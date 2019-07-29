Run DQN with extra-replay steps for a fixed target
```python
python run.py --id 0 --config-file experiment/config_files/valli/dqn_replay.json
```

Run DQN without extra replay steps 
```python
python run.py --id 0 --config-file experiment/config_files/valli/dqn_noreplay.json
```

Expected Sarsa with extra-replay steps for a fixed target
```python
python run.py --id 0 --config-file experiment/config_files/valli/sarsa_target_epsilon.json
```
