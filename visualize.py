import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("./scaling_law.csv")

# fig = px.line(x=df['num_params'], y=df['loss'], log_x=True)

fig = go.Figure()

# categories = df['series'].unique()

# for category in categories:
fig.add_trace(go.Scatter(
    x=df["num_params"],
    y=df['loss'],
    mode='lines+markers',  # Line with markers
))

fig.update_layout(
title={
    'text': "Loss vs. model parameters",
    'y':0.97,  # y-position of the title
    'x':0.5,  # x-position of the title (centered)
    'xanchor': 'center',
    'yanchor': 'top'
},
xaxis_title="Model parameters",
yaxis_title="Loss",
title_font=dict(size=20),  # Main title font size
xaxis=dict(title_font=dict(size=15)),  # X-axis label font size
yaxis=dict(title_font=dict(size=15),
            # range=[0, df['loss'].max() + 0.0001],
        ),  # Y-axis label font size
)

fig.update_xaxes(type="log")
# Show the figure
fig.show()
