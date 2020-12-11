from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import numpy as np
import time
import requests

try:
    options = webdriver.ChromeOptions()
    options.add_argument('--user-data-dir=default')
    options.add_argument("--window-size=1300,900")
    driver = webdriver.Chrome(options=options)
    driver.get("https://sleeper.app/leagues/599841620514373632")

    # Move to standings tab
    driver.find_element(By.ID, "standings-tab").click()
    time.sleep(1)

    page_source = driver.page_source
finally:
    driver.quit()

# Authoritative team ordering...
teams_correctly_ordered = teams_canonical = np.array(['daniel_reichl', 'mjgouvion', 'Torry28', 'mklink', 'ryanpdwyer',
       'JayRStorm', 'DavidAnd', 'br57', 'ajkahle', 'Parm123'],
      dtype='<U13')

n_teams = len(teams_canonical)

def extract_data(page_source):
    owner_names = {'@ryanpdwyer': 'Ryan',
 '@br57': 'Bryan',
 '@mklink': 'Mitch',
 '@DavidAnd': 'David',
 '@daniel_reichl': 'Daniel',
 '@ajkahle': 'AJ',
 '@JayRStorm': 'James',
 '@Parm123': 'Parm',
 '@mjgouvion': 'Mike',
 '@Torry28': 'Torry'}
    
    owners = {"@"+x: x for x in teams_canonical}
    


    soup = BeautifulSoup(page_source, 'lxml')
    matchups = soup.find('div', class_='league-matchups')
    team_ids = matchups.find_all('div', class_="team-name")
    score_els = matchups.find_all('div', class_="score")
    proj_els = matchups.find_all('div', class_="projections")

    teams = [owners[t.text] for t in team_ids]
    scores = np.array([float(el.text.replace('-', '0')) for el in score_els])
    projs = np.array([float(el.text) for el in proj_els])
    return (teams, scores, projs)



# rosterid_dict = {1: 'Daniel',
#  2: 'Mike',
#  3: 'Torry',
#  4: 'Mitch',
#  5: 'Ryan',
#  6: 'James',
#  7: 'David',
#  8: 'Bryan',
#  9: 'AJ',
#  10: 'Parm'}



teams, scores, projs = extract_data(page_source)
inds = [list(teams_correctly_ordered).index(t) for t in teams]

r = requests.get('https://api.sleeper.app/v1/league/599841620514373632/matchups/14')

# starters = {rosterid_dict[rr['roster_id']]: rr['starters'] for rr in r.json()}

# starters_out = {key: [players_out[v] for v in val if v in players_out] for key, val in starters.items()}

# bonus_points = {key: sum([10 if v != 'QB' else 22 for v in val]) for key, val in starters_out.items()}

# print(bonus_points)

# bonus_array = np.array([bonus_points[t] for t in teams_correctly_ordered])

scores_projs = np.zeros((n_teams, 2))
scores_projs[inds, 0] = scores
scores_projs[inds, 1] = projs

print(list(zip(teams_correctly_ordered, scores_projs[:, 0], scores_projs[:, 1])))

print(scores_projs)

np.savetxt('scores-proj-14.csv', scores_projs, fmt="%.1f")

