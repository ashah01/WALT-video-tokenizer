import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("./runs.csv")

fig = go.Figure()

categories = df['series'].unique()

for category in categories:
    category_data = df[df['series'] == category]
    fig.add_trace(go.Scatter(
        x=category_data['init_dim'] / (5 + 5*category_data['num_blocks']),
        y=category_data['loss'],
        mode='lines+markers',  # Line with markers
        name=category
    ))

fig.update_layout(
    title={
        'text': "Loss vs. Aspect ratio",
        'y':0.95,  # y-position of the title
        'x':0.5,  # x-position of the title (centered)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Aspect ratio",
    yaxis_title="Loss",
    title_font=dict(size=20),  # Main title font size
    xaxis=dict(title_font=dict(size=15)),  # X-axis label font size
    yaxis=dict(title_font=dict(size=15),
               range=[0, df['loss'].max() + 0.0001],),  # Y-axis label font size
)
# Show the figure
fig.show()
