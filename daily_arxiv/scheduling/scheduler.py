"""Persistent daily GPT filtering scheduler for machines without systemd/cron."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
import threading
from zoneinfo import ZoneInfo

from daily_arxiv.paths import DAILY_DIR, SCHEDULER_ENTRY

BASE = DAILY_DIR
UTC = timezone.utc


def next_daily(now, zone, at):
    hour, minute = map(int, at.split(':'))
    local = now.astimezone(ZoneInfo(zone))
    candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


def target_date(now, zone='Asia/Shanghai', day_lag=2):
    return (now.astimezone(ZoneInfo(zone)).date() - timedelta(days=day_lag)).isoformat()


def day_closed(day):
    return datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=UTC) + timedelta(days=1)


def record_outcome(state, now, started_at, end, code, stopping=False):
    state.update(last_exit_code=code, last_finished_at=now.isoformat())
    if stopping:
        state.update(status='interrupted', next_run_at=now.isoformat())
    elif code == 0 and started_at < day_closed(end):
        # At 07:00 Beijing, yesterday's UTC date is still open until 08:00.
        # Do not advance completion until a fresh query after the UTC boundary.
        follow_up = max(now, day_closed(end) + timedelta(minutes=30))
        state.update(status='waiting_for_day_close', next_run_at=follow_up.isoformat(), attempts=0)
    elif code == 0:
        state.update(status='waiting', last_completed_date=max(state.get('last_completed_date') or end, end), attempts=0,
                     next_run_at=next_daily(now, state['timezone'], state['at']).isoformat())
    else:
        scheduled = next_daily(now, state['timezone'], state['at'])
        # Keep recovering during the day instead of abandoning a date after
        # three transient fetch failures. Back off 1h, 1h, 2h, then at most 4h;
        # the next daily trigger always takes priority if it is sooner.
        delay_hours = 2 ** min(max(state['attempts'] - 2, 0), 2)
        retry = min(now + timedelta(hours=delay_hours), scheduled)
        state.update(status='retry_wait' if retry < scheduled else 'waiting_after_failure', next_run_at=retry.isoformat())


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def initial_state(args, now):
    from daily_arxiv.reporting.artifacts import load_completed_report
    completed = []
    model_dir = BASE / 'runs' / 'gpt-6-luna_runs'
    for report in (model_dir / 'hits_zh').glob('hits_zh_*.json'):
        day = report.stem.removeprefix('hits_zh_')
        try:
            datetime.strptime(day, '%Y-%m-%d')
            load_completed_report(model_dir, day)
            if day <= target_date(now, args.timezone, getattr(args, 'day_lag', 2)):
                completed.append(day)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {'timezone': args.timezone, 'at': args.at,
            'first_date': target_date(now, args.timezone, getattr(args, 'day_lag', 2)),
            'lookback_days': getattr(args, 'lookback_days', 7),
            'day_lag': getattr(args, 'day_lag', 2),
            'last_completed_date': max(completed) if completed else None,
            'next_run_at': next_daily(now, args.timezone, args.at).isoformat(),
            'attempt_day': None, 'attempts': 0, 'status': 'waiting'}


def date_range(state, now):
    end = target_date(now, state['timezone'], state.get('day_lag', 2))
    start = ((datetime.strptime(state['last_completed_date'], '%Y-%m-%d').date()
              + timedelta(days=1)).isoformat() if state['last_completed_date'] else state['first_date'])
    # Requery successful dates too: records can appear late with an older datestamp.
    overlap = (datetime.strptime(end, '%Y-%m-%d').date()
               - timedelta(days=state.get('lookback_days', 7) - 1)).isoformat()
    start = max(state['first_date'], min(start, overlap))
    return start, end


def command_for(args, start, end):
    return [sys.executable, str(BASE / 'run_daily_arxiv.py'),
            '--from_date', start, '--through_date', end,
            '--models', 'gpt-6-luna', '--workers', str(args.workers),
            '--codex_bin', args.codex_bin]


def running_pid(directory):
    try:
        pid = int((directory / 'scheduler.lock').read_text())
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(SCHEDULER_ENTRY).encode() in command and b'run' in command:
            return pid
    except (OSError, ValueError):
        pass
    return None


def run(args):
    directory = args.state_dir
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'scheduler.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Scheduler already running')
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        logger = logging.getLogger('arxiv-scheduler')
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(directory / 'scheduler.log', maxBytes=5_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(handler.formatter)
        logger.addHandler(console)
        path = directory / 'state.json'
        now = datetime.now(UTC)
        state = json.loads(path.read_text()) if path.exists() else initial_state(args, now)
        if (state.get('timezone'), state.get('at')) != (args.timezone, args.at):
            state.update(timezone=args.timezone, at=args.at,
                         next_run_at=next_daily(now, args.timezone, args.at).isoformat())
        if state['status'] in ('running', 'interrupted'):
            state['next_run_at'] = now.isoformat()
        state['day_lag'] = args.day_lag
        state['lookback_days'] = args.lookback_days
        save_state(path, state)
        stopping, child = False, None
        stopped = threading.Event()

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            stopped.set()
            if child and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        logger.info('Started PID %s; daily %s %s; next %s', os.getpid(), args.timezone, args.at, state['next_run_at'])
        environment = dict(os.environ)
        # Codex's shebang uses env node; cron-like environments need the node directory.
        environment['PATH'] = str(Path(args.codex_bin).parent) + os.pathsep + environment.get('PATH', '')
        while not stopping:
            now = datetime.now(UTC)
            if getattr(args, 'publish_site', False) and state.get('site_pending_through'):
                retry_at = datetime.fromisoformat(state.get('site_retry_at', now.isoformat()))
                if now >= retry_at:
                    try:
                        result = subprocess.run(
                            [sys.executable, '-m', 'daily_arxiv.reporting.site',
                             '--through-date', state['site_pending_through']],
                            cwd=BASE.parent, env=environment, capture_output=True,
                            text=True, timeout=300)
                        if result.returncode:
                            raise RuntimeError(result.stderr[-2000:])
                        state.update(site_last_published_through=state.pop('site_pending_through'),
                                     site_last_published_at=datetime.now(UTC).isoformat())
                        state.pop('site_retry_at', None)
                        state.pop('site_last_error', None)
                        logger.info('Site publication: %s', result.stdout.strip())
                    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                        state.update(site_retry_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                                     site_last_error=str(exc))
                        logger.error('Site publication failed; retry in one hour: %s', exc)
                    save_state(path, state)
                    now = datetime.now(UTC)
            due = datetime.fromisoformat(state['next_run_at'])
            if now < due:
                stopped.wait(min(30, (due - now).total_seconds()))
                continue
            start, end = date_range(state, now)
            if start > end:
                state.update(status='waiting', next_run_at=next_daily(now, args.timezone, args.at).isoformat())
                save_state(path, state)
                continue
            local_day = now.astimezone(ZoneInfo(args.timezone)).date().isoformat()
            if state['attempt_day'] != local_day:
                state.update(attempt_day=local_day, attempts=0)
            state['attempts'] += 1
            started_at = now
            logfile = directory / f'job_{end}_{now.strftime("%Y%m%dT%H%M%SZ")}.log'
            state.update(status='running', active_from=start, active_through=end,
                         last_job_log=str(logfile), last_started_at=now.isoformat())
            save_state(path, state)
            logger.info('Run %s through %s; attempt %s; log %s', start, end, state['attempts'], logfile)
            code = 1
            try:
                with logfile.open('a') as output:
                    child = subprocess.Popen(command_for(args, start, end), cwd=BASE.parent,
                                             env=environment, stdin=subprocess.DEVNULL,
                                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None:
                        if stopping:
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                        else:
                            stopped.wait(1)
                    code = child.returncode
            except OSError:
                logger.exception('Could not run filtering command')
            finally:
                child = None
            now = datetime.now(UTC)
            record_outcome(state, now, started_at, end, code, stopping)
            if getattr(args, 'publish_site', False) and code == 0 and not stopping and state['status'] == 'waiting':
                state.update(site_pending_through=end, site_retry_at=now.isoformat())
            save_state(path, state)
            logger.info('Exit %s; status %s; last completed %s; next %s', code, state['status'], state['last_completed_date'], state['next_run_at'])
        logger.info('Scheduler stopped')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'run', 'status', 'stop', 'plan'))
    parser.add_argument('--at', default='01:00', help='Daily local HH:MM')
    parser.add_argument('--timezone', default='Asia/Shanghai')
    parser.add_argument('--day_lag', type=int, default=2, help='Metadata date lag in local calendar days (2: day before yesterday)')
    parser.add_argument('--lookback_days', type=int, default=7, help='Recheck recent metadata dates to catch late records')
    parser.add_argument('--session', default='arxiv-daily', help='tmux session name for start')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent GPT requests (default: 1)')
    parser.add_argument('--publish-site', action='store_true', help='Publish completed reports to the gh-pages branch; retry uploads separately')
    parser.add_argument('--codex_bin', default='codex')
    parser.add_argument('--state_dir', type=Path, default=BASE / 'runs' / '.scheduler')
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.at, '%H:%M')
        ZoneInfo(args.timezone)
        if args.workers < 1 or args.lookback_days < 1 or args.day_lag < 1:
            raise ValueError('workers, lookback_days, and day_lag must be positive')
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.state_dir = args.state_dir.resolve()
    if args.command in ('start', 'run', 'plan'):
        executable = shutil.which(args.codex_bin)
        if not executable:
            parser.error('Codex executable not found')
        args.codex_bin = str(Path(executable).resolve()) if not Path(executable).is_symlink() else str(Path(executable).absolute())
    if args.command == 'status':
        path = args.state_dir / 'state.json'
        print(json.dumps({'pid': running_pid(args.state_dir), 'state': json.loads(path.read_text()) if path.exists() else None}, ensure_ascii=False, indent=2))
    elif args.command == 'stop':
        pid = running_pid(args.state_dir)
        if pid:
            os.kill(pid, signal.SIGTERM)
            print(f'Stop requested for PID {pid}')
        else:
            print('Scheduler is not running')
    elif args.command == 'plan':
        print(json.dumps(initial_state(args, datetime.now(UTC)), ensure_ascii=False, indent=2))
    elif args.command == 'run':
        run(args)
    else:
        if running_pid(args.state_dir):
            print(f'Already running: PID {running_pid(args.state_dir)}')
            return
        args.state_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(SCHEDULER_ENTRY), 'run', '--at', args.at,
                   '--timezone', args.timezone, '--workers', str(args.workers),
                   '--lookback_days', str(args.lookback_days), '--day_lag', str(args.day_lag),
                   '--codex_bin', args.codex_bin, '--state_dir', str(args.state_dir)]
        if args.publish_site:
            command.append('--publish-site')
        if not shutil.which('tmux'):
            raise RuntimeError('tmux executable not found')
        if subprocess.run(['tmux', 'has-session', '-t', '=' + args.session], capture_output=True).returncode == 0:
            raise RuntimeError(f'tmux session {args.session} already exists; choose another --session')
        shell_command = 'exec ' + shlex.join(command) + ' 2>>' + shlex.quote(str(args.state_dir / 'daemon.log'))
        subprocess.run(['tmux', 'new-session', '-d', '-s', args.session,
                        '-c', str(BASE.parent), shell_command], check=True)
        for _ in range(50):
            if subprocess.run(['tmux', 'has-session', '-t', '=' + args.session], capture_output=True).returncode:
                raise RuntimeError(f'Scheduler startup failed; see {args.state_dir / "daemon.log"}')
            if running_pid(args.state_dir) and (args.state_dir / 'state.json').exists():
                print(f'Scheduler started in tmux {args.session}: PID {running_pid(args.state_dir)}; daily {args.at} {args.timezone}')
                return
            time.sleep(0.1)
        raise RuntimeError('Scheduler startup timed out')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
