import pandas as pd
import json

with open('my_plotting/trainer_state.json', encoding="utf-8") as f:
    data = json.load(f)
step_data = {}
for entry in data["log_history"]:
    step = entry['step']
    if step not in step_data:
        step_data[step] = {'train_loss': None, 'eval_loss': None}
    
    if 'loss' in entry:
        step_data[step]['train_loss']=(entry['loss'])
    if 'eval_loss' in entry:
        step_data[step]['eval_loss']=(entry['eval_loss'])

df = pd.DataFrame.from_dict(step_data, orient='index').reset_index()
df.columns = ['step', 'train_loss', 'eval_loss']
excel_path = 'my_plotting/loss_eval_loss.xlsx'
df.to_excel(excel_path, index=False)

