import plotly.graph_objects as go

# Define the nodes and links
nodes = [
    {"label": "Failure Modes"},
    {"label": "Pruning error"},
    {"label": "Correct Pruning"},
    {"label": "Semantic Context"},
    {"label": "Hallucinate Nodes/Edges"},
    {"label": "Correct TMI"},
    {"label": "Correct Partial"},
    {"label": "Eval Error"},
]

links = [
    {"source": 0, "target": 1, "value": 1},
    {"source": 0, "target": 2, "value": 26},
    {"source": 2, "target": 3, "value": 9},
    {"source": 2, "target": 4, "value": 6},
    {"source": 2, "target": 5, "value": 6},
    {"source": 2, "target": 6, "value": 3},
    {"source": 2, "target": 7, "value": 2},
]

# Create the sankey diagram
fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[node["label"] for node in nodes],
            ),
            link=dict(
                source=[link["source"] for link in links],
                target=[link["target"] for link in links],
                value=[link["value"] for link in links],
            ),
        )
    ]
)

# fig.update_layout(title_text="Failure", font_size=10)
# Save the figure as a PNG image
fig.write_image("failure_modes.svg")
