from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio

app = Flask(__name__)

# Load your dataset
nfl_main_df = pd.read_csv("nfl_main.csv")

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/game_dashboard', methods=['GET'])
def game_dashboard():
    week = request.args.get('week', default=1, type=int)
    season = request.args.get('season', default=2024, type=int)
    team = request.args.get('team', default='PHI', type=str)
    stats = request.args.getlist('stats')
    
    week_df = nfl_main_df[(nfl_main_df["week"] == week) & (nfl_main_df["season"] == season)]
    if week_df.empty:
        return "No data for the selected filters.", 404

    try:
        game_id = week_df[week_df["team"] == team]["game_id"].values[0]
        game_df = nfl_main_df[nfl_main_df["game_id"] == game_id]

        num_stats = len(stats)
        ncols = 2
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
                fig.add_annotation(
                    x=0.8, y=0.8,
                    text="No Data",
                    showarrow=False,
                    xref=f"x{i+1}",
                    yref=f"y{i+1}",
                    font=dict(color="red"),
                    row=row, col=col
                )

        fig.update_layout(
            height=400 * nrows,
            width=800,
            title_text="Game Statistics",
            showlegend=False
        )
        fig.update_xaxes(tickangle=-70)

        plot_data = pio.to_json(fig)
        return render_template('dashboard.html', plot_data=plot_data)

    except IndexError:
        return "No matching game data for the selected week, season, and team.", 404

@app.route('/season_dashboard', methods=['GET'])
def season_dashboard():
    season = request.args.get('season', default=2024, type=int)
    stats = request.args.getlist('stats')
    
    season_df = nfl_main_df[(nfl_main_df["season"] == season)].groupby(["player_display_name", "team"])[stats].sum().reset_index()
    if season_df.empty:
        return "No data for the selected filters.", 404

    try:
        num_stats = len(stats)
        ncols = 2
        nrows = int(np.ceil(num_stats / ncols))

        fig = sp.make_subplots(rows=nrows, cols=ncols, subplot_titles=stats)

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
                fig.add_annotation(
                    x=0.8, y=0.8,
                    text="No Data",
                    showarrow=False,
                    xref=f"x{i+1}",
                    yref=f"y{i+1}",
                    font=dict(color="red"),
                    row=row, col=col
                )

        fig.update_layout(
            height=400 * nrows,
            width=800,
            title_text="Season Statistics",
            showlegend=False
        )
        fig.update_xaxes(tickangle=-70)

        plot_data = pio.to_json(fig)
        return render_template('season_dashboard.html', plot_data=plot_data)

    except IndexError:
        return "No matching data for the selected season.", 404

if __name__ == '__main__':
    app.run(debug=True)
