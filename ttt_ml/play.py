from tqdm import tqdm

from ttt_ml.board import EPISODES

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    done = False
    #while not done:
