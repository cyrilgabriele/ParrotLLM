# TMUX session workflow

## 1. Create a TMUX session with the experiment name

Replace `<experiment_name>` with your actual experiment name:

```bash
tmux new -s <experiment_name>
```

Example:

```bash
tmux new -s my_experiment
```

This creates and immediately enters a new TMUX session named `my_experiment`.

## 2. Test whether you are in the assumed session

Run:

```bash
tmux display-message -p '#S'
```

This prints the name of the current TMUX session.

Example output:

```bash
my_experiment
```

If the printed session name matches the expected experiment name, then you are in the correct session.

## 3. Optionally detach from the session

To leave the TMUX session while keeping it running in the background, press:

```text
Ctrl-b d
```

Explanation:

- Press `Ctrl-b`
- Release both keys
- Then press `d`

This detaches you from the session without stopping the processes running inside it.

## 4. Attach back to the session later

To re-attach to the session, run:

```bash
tmux attach -t <experiment_name>
```

Example:

```bash
tmux attach -t my_experiment
```

This returns you to the running TMUX session.
