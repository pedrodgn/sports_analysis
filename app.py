from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import os
import numpy as np

app = Flask(__name__)

# Load the data
csv_path = os.path.join(os.getcwd(), 'nfl_main.csv')
nfl_main_df = pd.read_csv(csv_path)

team_colors = {
    'ARI': 'rgb(151, 35, 63)', 'ATL': 'rgb(0, 0, 0)', 'BAL': 'rgb(26, 25, 95)', 'BUF': 'rgb(0, 51, 141)',
    'CAR': 'rgb(0, 133, 202)', 'CHI': 'rgb(11, 22, 42)', 'CIN': 'rgb(255, 60, 0)', 'CLE': 'rgb(49, 29, 0)',
    'DAL': 'rgb(0, 34, 68)', 'DEN': 'rgb(0, 34, 68)', 'DET': 'rgb(0, 118, 182)', 'GB': 'rgb(24, 48, 40)',
    'HOU': 'rgb(3, 32, 47)', 'IND': 'rgb(0, 44, 95)', 'JAX': 'rgb(0, 103, 120)', 'KC': 'rgb(227, 24, 55)',
    'LA': 'rgb(0, 53, 148)', 'LAC': 'rgb(0, 128, 198)', 'LV': 'rgb(0, 0, 0)', 'MIA': 'rgb(0, 142, 151)',
    'MIN': 'rgb(79, 38, 131)', 'NE': 'rgb(0, 34, 68)', 'NO': 'rgb(211, 188, 141)', 'NYG': 'rgb(1, 35, 82)',
    'NYJ': 'rgb(18, 87, 64)', 'PHI': 'rgb(0, 76, 84)', 'PIT': 'rgb(0, 0, 0)', 'SEA': 'rgb(0, 21, 50)',
    'SF': 'rgb(170, 0, 0)', 'TB': 'rgb(213, 10, 10)', 'TEN': 'rgb(68, 149, 209)', 'WAS': 'rgb(63, 16, 16)'
}

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Game Dashboard Route
@app.route('/game-dashboard', methods=['GET', 'POST'])
def game_dashboard():
    weeks = list(nfl_main_df["week"].sort_values().unique())
    seasons = list(nfl_main_df["season"].sort_values().unique())
    teams = list(nfl_main_df["team"].sort_values().unique())
    stats_options = [
        'completions', 'attempts',
       'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards',
       'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
       'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
       'passing_2pt_conversions', 'pacr', 'dakota', 'carries', 'rushing_yards',
       'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
       'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
       'receptions', 'targets', 'receiving_yards', 'receiving_tds',
       'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
       'receiving_yards_after_catch', 'receiving_first_downs'
    ]

    if request.method == 'POST':
        week = int(request.form['week'])
        season = int(request.form['season'])
        team = request.form['team']
        stats = request.form.getlist('stats')

        # Generate plot based on user input
        return render_template('game_dashboard.html', plot=create_game_plot(week, season, team, stats), weeks=weeks, seasons=seasons, teams=teams, stats_options=stats_options)

    return render_template('game_dashboard.html', weeks=weeks, seasons=seasons, teams=teams, stats_options=stats_options)

# Season Dashboard Route
@app.route('/season-dashboard', methods=['GET', 'POST'])
def season_dashboard():
    seasons = list(nfl_main_df["season"].sort_values().unique())
    stats_options = [
        'completions', 'attempts',
       'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards',
       'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
       'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
       'passing_2pt_conversions', 'pacr', 'dakota', 'carries', 'rushing_yards',
       'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
       'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
       'receptions', 'targets', 'receiving_yards', 'receiving_tds',
       'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
       'receiving_yards_after_catch', 'receiving_first_downs'
    ]

    if request.method == 'POST':
        season = int(request.form['season'])
        stats = request.form.getlist('stats')

        # Generate plot based on user input
        return render_template('season_dashboard.html', plot=create_season_plot(season, stats), seasons=seasons, stats_options=stats_options)

    return render_template('season_dashboard.html', seasons=seasons, stats_options=stats_options)

# Plot creation for the game dashboard
def create_game_plot(week, season, team, stats):
	
    if not stats:
        return "Please select at least one statistic."	
	
    week_df = nfl_main_df[(nfl_main_df["week"] == week) & (nfl_main_df["season"] == season)]
    if week_df.empty:
        return "No data for the selected filters."

    game_id = week_df[week_df["team"] == team]["game_id"].values[0]
    game_df = nfl_main_df[nfl_main_df["game_id"] == game_id]

    num_stats = len(stats)
    ncols = 3
    nrows = int(np.ceil(num_stats / ncols))

    fig = sp.make_subplots(rows=nrows, cols=ncols, subplot_titles=stats)

    for i, s in enumerate(stats):
        filtered_data = game_df[game_df[s] > 0].sort_values(s, ascending=False)
        if not filtered_data.empty:
            colors = [team_colors.get(t, 'grey') for t in filtered_data['team']]
            trace = go.Bar(
                        x=filtered_data["player_display_name"],
                        y=filtered_data[s],
                        name=s,
                        marker=dict(color=colors),
                        text=filtered_data["team"],
                        hoverinfo="text+y"
                    )
            row = i // ncols + 1
            col = i % ncols + 1
            fig.add_trace(trace, row=row, col=col)
        else:
            row = i // ncols + 1
            col = i % ncols + 1
            fig.add_annotation(x=0.5, y=0.5, text="No Data", showarrow=False, xref=f"x{i+1}", yref=f"y{i+1}", font=dict(color="red"))

    fig.update_layout(
        height=400 * nrows, 
        width=1000,  # Increased width to accommodate larger subplots
        title_text=f"Game Dashboard for {team} in Week {week} - {season}"
    )

    # Update x-axis properties
    for i in range(num_stats):
        fig.update_xaxes(tickangle=70, row=(i // ncols) + 1, col=(i % ncols) + 1)

    return fig.to_html(full_html=False)

# Plot creation for the season dashboard
def create_season_plot(season, stats):

    if not stats:
        return "Please select at least one statistic."
    season_df = nfl_main_df[nfl_main_df["season"] == season].groupby(["player_display_name", "team"])[stats].sum().reset_index()
    if season_df.empty:
        return "No data for the selected filters."

    num_stats = len(stats)
    ncols = 3
    nrows = int(np.ceil(num_stats / ncols))

    fig = sp.make_subplots(rows=nrows, cols=ncols, subplot_titles=stats, horizontal_spacing=0.15, vertical_spacing=0.2)

    for i, s in enumerate(stats):
        filtered_data = season_df[season_df[s] > 0].sort_values(s, ascending=False)[:10]
        if not filtered_data.empty:
            colors = [team_colors.get(t, 'grey') for t in filtered_data['team']]
            trace = go.Bar(
                        x=filtered_data["player_display_name"],
                        y=filtered_data[s],
                        name=s,
                        marker=dict(color=colors),
                        text=filtered_data["team"],
                        hoverinfo="text+y"
                    )
            row = i // ncols + 1
            col = i % ncols + 1
            fig.add_trace(trace, row=row, col=col)
        else:
            row = i // ncols + 1
            col = i % ncols + 1
            fig.add_annotation(x=0.5, y=0.5, text="No Data", showarrow=False, xref=f"x{i+1}", yref=f"y{i+1}", font=dict(color="red"))

    fig.update_layout(
        height=420 * nrows,
        width=1200,  # Increased width to accommodate larger subplots
        title_text=f"Season Dashboard for {season}"
    )

    # Update x-axis properties
    for i in range(num_stats):
        fig.update_xaxes(tickangle=70, row=(i // ncols) + 1, col=(i % ncols) + 1)

    return fig.to_html(full_html=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
