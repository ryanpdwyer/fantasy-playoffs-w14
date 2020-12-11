import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import requests
import datetime
from collections import OrderedDict, defaultdict, Counter

currentYear = datetime.datetime.today().year

def gamma(mean, shape):
    return np.random.default_rng().gamma(mean/5, 5, shape)

def groupby(d):
    res = defaultdict(list)
    for key, val in sorted(d.items()):
        res[val].append(key)
    return res

def get_opponents(d):
    x = np.array(list(groupby(d).values())) - 1 # Subtract 1 so we have indices
    out = np.zeros(x.size, dtype=int)
    for i in range(x.shape[0]):
        out[x[i, 0]] = x[i, 1]
        out[x[i, 1]] = x[i, 0]
    return out 

def get_opponents_no_off_one(d):
    x = np.array(list(groupby(d).values())) # Subtract 1 so we have indices
    out = np.zeros(x.size, dtype=int)
    for i in range(x.shape[0]):
        out[x[i, 0]] = x[i, 1]
        out[x[i, 1]] = x[i, 0]
    return out 

def get_id(url_or_id):
    possible_ids = [x for x in url_or_id.split("/") if x.isdigit()]
    return possible_ids[0]

@st.cache
def get_league(id):
    r = requests.get("https://api.sleeper.app/v1/league/"+id)
    return r.json()

@st.cache
def get_users(id):
    return requests.get("https://api.sleeper.app/v1/league/{}/users".format(id)).json()

@st.cache
def get_rosters(id):
    return requests.get("https://api.sleeper.app/v1/league/{}/rosters".format(id)).json()

@st.cache
def get_matchups(id, week):
    return requests.get("https://api.sleeper.app/v1/league/{}/matchups/{}".format(id, week)).json()

def get_matchups_live(id, week):
    return requests.get("https://api.sleeper.app/v1/league/{}/matchups/{}".format(id, week)).json()


def filter_rosters(rosters):
    return {r['roster_id']: dict(wins=r['settings']["wins"], losses=r['settings']['losses'], ties=r['settings']['ties'],
            pts=(int(r['settings']['fpts'])+int(r['settings']['fpts_decimal'])/100),
            division=r['settings']['division']) for r in rosters}

def intx(x):
    return int(x) if x != '' else x

def rotisserie(pts):
    rotis = np.argsort(pts)
    n_games, n_teams = pts.shape
    rr = np.zeros_like(pts)
    for i in range(n_games):
        rr[i, rotis[i]] = np.arange(n_teams)
    
    return rr

@st.cache
def simulate_remaining_weeks(games_left, n_teams, N, pts_regress, stdev, future_opponents):
    pts_unplayed = np.zeros((games_left, n_teams, N))
    wins_unplayed = np.zeros((games_left, n_teams, N), dtype=bool)
    rotis_unplayed = np.zeros((games_left, n_teams, N), dtype=int)

    for i in range(games_left):
        pts_unplayed[i] = rng.normal(scale=stdev,
                        loc=pts_regress.reshape((-1, 1)), size=(n_teams, N))
        rotis_unplayed[i] = rotisserie(pts_unplayed[i].T).T
        wins_unplayed[i] = pts_unplayed[i] > pts_unplayed[i][future_opponents[i]]

    
    return pts_unplayed, wins_unplayed, rotis_unplayed


@st.cache
def simulate_round_one(n_teams, N, pts, bonus, projections, opponents):
    pass

def playoffs_fast(wins_, pts_regress, seeds, std):
    playoff_inds = np.where(seeds <= n_playoff_teams)[0]


    # Just the playoff teams
    pwins = wins_[playoff_inds]
    ppts_regress = pts_regress[playoff_inds]
    pseeds = list(seeds[playoff_inds])


    playoffPts = np.sum(np.random.randn(n_playoff_teams, 2)*std, axis=1) + ppts_regress*2

    finalists = playoffPts[pwins]

    finish_order = np.argsort(finalists)[::-1]


    yyy = np.ones_like(seeds, dtype=int)*(n_playoff_teams+1)
    yyy[playoff_inds[pwins][finish_order]] = np.arange(1,4)

    return yyy

@st.cache
def makeAllPlayoffResults(overall_wins, pts_regress, seeds, std):
    return np.array(
    [playoffs_fast(wins_, pts_regress, seeds, std) for wins_ in overall_wins])


def analyzePlayoffResults(playoffResults, teams):
    winsCount = {team: Counter(play) for team, play in zip(teams,
                                                    playoffResults.T)}
    dfPR = pd.DataFrame.from_dict(winsCount, orient='index').fillna(0)/len(playoffResults)*100
    dfPR.sort_values(1, ascending=False, inplace=True)
    dfPRO = dfPR.loc[:, [1, 2, 3]].rename(columns={1: "Champion", 2: "2nd", 3: "3rd" })
    dfPRO['Cash'] = (dfPRO["Champion"]*350 + dfPRO["2nd"]*100 + dfPRO["3rd"]*50)/ 100
    dfPRO.round(1)
    try:
        del dfPRO['avgSeed']
    except:
        pass
    return dfPRO

# st.write('''<div style='height: 200px; background-image: url("https://lh5.googleusercontent.com/GAmGcsk5dPuofSN8eXAYinlFC8lxIQdvynYW7CgyoGRhOy_em36mmJ1pNtpchiz7Es-q86OUjA=w16383");'></div>''', unsafe_allow_html=True)

st.title("Toolshed Championship Odds")
league_website = "Sleeper"
url = "599841620514373632"
season_weeks = 14
game_vs_league_median = False


id = get_id(url)

# Sleeper specific
rjson = get_league(id)
users = get_users(id)
rosters = get_rosters(id)

owners = OrderedDict((user['user_id'], user['display_name']) for user in users)
# teams = list(owners.values())
roster_owner = OrderedDict((r['roster_id'], r['owner_id']) for r in rosters)
roster_display = OrderedDict((key, owners[val]) for key, val in roster_owner.items())

## Everything needs this list of canonical teams, number of teams
## Number of playoff teams
teams_canonical = np.array(list(roster_display.values()))
n_teams = len(teams_canonical)


n_playoff_teams = rjson['settings']['playoff_teams']


# Sleeper specific
rr = filter_rosters(rosters)
rr_df = pd.DataFrame.from_dict({owners[roster_owner[key]]: val for key, val in rr.items()},
                orient='index')

divisions = np.zeros(n_teams, dtype=int)
for id_, val in rr.items():
    divisions[id_-1] = val['division']


games = (rr_df['wins'].iloc[0] + rr_df['losses'].iloc[0] + rr_df['ties'].iloc[0])
week = games + 1
# Contains wins, losses, points, almost everything
# we need on a week-by-week basis

matchups = [get_matchups(id, w) if w != week else get_matchups_live(id, w) for w in range(1, season_weeks+1)]


# Using roster_id as the canonical ordering
pts_dict = [{x['roster_id']: x['points'] for x in y} for y in matchups]
weekly_matchups = [{x['roster_id']: x['matchup_id'] for x in y} for y in matchups]
weekly_opponents = np.array([get_opponents(w) for w in weekly_matchups])
pts = np.array([list(p.values()) for p in pts_dict], dtype=float)

    # Once I have the pts array, and the matchups list,
    # I can generate everything else from scratch! ----



    # This ends the sleeper specific part of the code

def make_opp_arr(x):
    x = np.array(x)
    out = np.zeros(x.size, dtype=int)
    for i in range(x.shape[0]):
        out[x[i, 0]] = x[i, 1]
        out[x[i, 1]] = x[i, 0]
    return out 


if url != "" and season_weeks != '':
    # This stuff is largely reproducible...
    pts_played = pts[:games]
    wins_played = np.array([pts_row > pts_row[opp] for pts_row, opp in zip(pts_played, weekly_opponents[:games])])
    total_wins = wins_played.sum(axis=0)

    rotis = rotisserie(pts_played)
    rotis_win_pct = rotis.mean(axis=0)/(n_teams - 1)
    rotis_wins = rotis >= int(n_teams/2)

    if game_vs_league_median:
        total_wins += rotis_wins.sum(axis=0)
    
    seeds = np.array([1,2,6,4,5,7,8,10,9,3], dtype=int) # Authoritative seeding

    df_standings = pd.DataFrame(np.c_[seeds, total_wins, pts.sum(axis=0), rotis_win_pct], index=teams_canonical,
                            columns=['Seed', 'Wins', 'Pts', 'Rotis. Win %'])
    
    
    df_standings = pd.DataFrame(np.c_[seeds, total_wins, pts.sum(axis=0), rotis_win_pct], index=teams_canonical,
                            columns=['Seed', 'Wins', 'Pts', 'Rotis. Win %'])
    
    df_standings.sort_values('Seed', inplace=True)

    # This is not sleeper specific...
    ptsAvg = pts_played.mean()
    bonus = np.zeros(n_teams)
    bonus[0] = 15.7
    bonus[1] = 7.9
    pts_regress = pts_played.mean(axis=0)*0.5 + ptsAvg*0.5
    std = pts_played.std()

    scale_factor = 0.953
    # Custom
    scores_proj_w1 = np.loadtxt("scores-proj-14.csv")
    pts_when_proj_w1 = scores_proj_w1[:, 0]
    pts_left_when_proj = (scores_proj_w1[:, 1] - pts_when_proj_w1) * scale_factor
    # Make sure that there is always at least 1 more point left in projection, just so that there's some uncertainty.
    pts_left_when_proj = np.where((pts_left_when_proj > 0) & (pts_left_when_proj < 1), 1, pts_left_when_proj)


    
    # 
    future_matchups = [np.array(list(groupby(m).values()))[:3]-1 for m in weekly_matchups[games:]]
    future_opponents = weekly_opponents[games:]
    
    #  games:
    games_left = 1 # Do one week at a time here...


    # Simulation
    N = 10000

    current_pts = pts[-1, :]
    pts_proj = np.r_[[gamma(pts, N) for pts in pts_left_when_proj]].T + current_pts + bonus

    wins_proj = pts_proj > pts_proj[:, future_opponents.flatten()]

    pts_unplayed = pts_proj.T.reshape((1, n_teams, N))
    wins_unplayed = wins_proj.T.reshape((1, n_teams, N))


    # pts_unplayed
    # wins_unplayed

    overall_pts = (pts_played.sum(axis=0).reshape((-1,1)) + pts_unplayed.sum(axis=0)).T

    playoffResults = makeAllPlayoffResults(wins_proj, pts_regress, seeds, std)

    inds = np.arange(N, dtype=int)


    # st.write("Projected outcomes for each game:")
    slots = [st.empty() for _ in range(games_left*2)]

    st.write("Use the buttons to see how championship chances change depending on the results of each game.")

    # Make buttons:
    n_cols = 3
    buttons = []
    for j, match_ in enumerate(future_matchups):
        cols = st.beta_columns(n_cols) 
        weekly_buttons = []
        for i, x in enumerate(match_):
            weekly_buttons.append(cols[i].radio(label='Winner',
                options=['Any', teams_canonical[x[0]], teams_canonical[x[1]]]
            )
        )
        buttons.append(weekly_buttons)
    

    # Filter simulations:
    for i, weekly_buttons in enumerate(buttons):
        for button in weekly_buttons:
            if button != 'Any':
                inds_match = np.nonzero(wins_unplayed[i, list(teams_canonical).index(button), :])
                inds = np.intersect1d(inds, inds_match, assume_unique=True)


    # Print weekly matchup projections
    for i, match_ in enumerate(future_matchups):
        slots[i*2].subheader("First Round Playoff Games")
        dfWeek = pd.DataFrame(np.c_[
                wins_unplayed[i, :, inds].mean(axis=0)*100,
                pts_unplayed[i, :, inds].mean(axis=0), 
                bonus + pts[week+i-1, :], 
                pts[week+i-1, :],
                bonus],
            index=teams_canonical, columns=['Win Prob', 'Proj. Total', 'Total', 'Pts', 'Bonus'])
        slots[i*2+1].dataframe(dfWeek.iloc[match_.flatten()].style.format("{:.1f}")\
        .format("{:.0f}", subset=["Win Prob"])\
        .background_gradient(cmap='RdBu_r', low=1.25, high=1.25, axis=0, subset=['Win Prob'])
        )

    

    dfPRO = analyzePlayoffResults(playoffResults[inds], teams_canonical).loc[teams_canonical[seeds <= n_playoff_teams]]
    dfPRO.sort_values("Champion", ascending=False, inplace=True)
    

    st.subheader("Championship Chances")
    st.dataframe(dfPRO.style.format("{:.0f}")\
        .format("${:.0f}", subset=['Cash'])\
        .background_gradient(cmap='Greens', low=0.0, high=0.7))
    


    st.subheader("Final Regular Season Standings")

    # This should be generated from scratch...
    st.dataframe(df_standings.style.format("{:.0f}", subset=['Seed'])\
                            .format("{:.1f}", subset=['Pts', 'Wins'])\
                            .format("{:.3f}", subset=['Rotis. Win %'])\
                            .background_gradient(cmap='RdBu_r', low=1.25, high=1.25, subset=['Wins', 'Pts', 'Rotis. Win %']))

    

    # Choose playoff teams...
    # Bracket...
    # Byes?

