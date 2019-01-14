import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np

import pickle

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN, SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

app = dash.Dash(__name__)
server = app.server

#################################################### NAVIGATION BAR MENU ####################################################
####################################################                     ####################################################

# Configure navbar menu
nav_menu = html.Div([
    html.Ul([
            # html.Li([dcc.Link('Introduction', href='intro')]),
            html.Li([dcc.Link('EDA', href='/eda')]),
            html.Li([dcc.Link('Modeling', href='/modeling')])
            ], className='nav navbar-nav')
], className='navbar navbar-default navbar-static-top')

# Content to be rendered in this element
content = html.Div(id='page-content')

# Define layout
app.layout = html.Div([
    # represents the URL bar, doesn't render anything
    dcc.Location(id='url', refresh=False),
    nav_menu,
    content
])

#################################################### INTRO ####################################################
####################################################       ####################################################

# Data for Intro Page
df_post_eda = pd.read_csv('loan_post_eda.csv', low_memory=False)

# Split the Features into Numerical and Non Numerical for charting purposes
features_numerical = list(df_post_eda.dtypes[df_post_eda.dtypes != 'object'].index)

features_nonnumerc = list(df_post_eda.dtypes[df_post_eda.dtypes == 'object'].index)
features_nonnumerc.remove('title')
features_nonnumerc.remove('emp_title')
features_nonnumerc.remove('desc')
features_nonnumerc.remove('loan_status')

features_nlp = ['title', 'emp_title', 'desc']

# Layout for Intro Page
# layout_page_1 = html.Div([
#     html.H2('Introduction to Lending Club Data'),
#     html.H3('Numerical Data'),
#     dt.DataTable(
#         id='table',
#         columns=[{"name": i, "id": i} for i in features_numerical],
#         data=df_post_eda[features_numerical].sample(10).to_dict("rows")
#     ),
#     html.H3('Non-Numerical Data'),
#     dt.DataTable(
#         id='table',
#         columns=[{"name": i, "id": i} for i in features_nonnumerc],
#         data=df_post_eda[features_nonnumerc].sample(10).to_dict("rows")
#     ),
#     html.H3('Unstructured Data'),
#     dt.DataTable(
#         id='table',
#         columns=[{"name": i, "id": i} for i in features_nlp],
#         data=df_post_eda[features_nlp].sample(10).to_dict("rows")
#     ),
# ])

layout_page_1 = {}

#################################################### EDA ####################################################
####################################################     ####################################################

# Data for EDA Page
# Use the same data from the Intro Page
x_loan_amnt = np.array([df_post_eda[df_post_eda['loan_status'] == 'Late (31-120 days)']['loan_amnt'].T,
                        df_post_eda[df_post_eda['loan_status'] == 'Charged Off']['loan_amnt'].T,
                        df_post_eda[df_post_eda['loan_status'] == 'Default']['loan_amnt'].T])

trace_grade = go.Box(
    y=df_post_eda['loan_amnt'],
    x=df_post_eda.sort_values(by='grade', ascending=True)['grade'],
    name='Grade',
    marker=dict(
        color='#3D9970'
    )
)
    
data = [trace_grade]

layout = go.Layout(
    yaxis=dict(
        title='Loan Amount',
        zeroline=False
    ),
    boxmode='group'
)

# Layout for the EDA Page
layout_page_2 = html.Div([
    html.H3('Number of Loans by Loan Status'),
    dcc.Graph(
        id='loans_by_status',
        figure={
            'data': [
                {
                    'x': df_post_eda[df_post_eda['loan_status']=='Default']['loan_status'].value_counts(),
                    'text': 'Default',
                    'name': 'Default',
                    'type': 'bar',
                },
                {
                    'x': df_post_eda[df_post_eda['loan_status']=='Charged Off']['loan_status'].value_counts(),
                    'text': 'Charged Off',
                    'name': 'Charged Off',
                    'type': 'bar'
                },
                {
                    'x': df_post_eda[df_post_eda['loan_status']=='Late (31-120 days)']['loan_status'].value_counts(),
                    'text': 'Late (1 to 4 months)',
                    'name': 'Late (1 to 4 months)',
                    'type': 'bar'
                },
                {
                    'x': df_post_eda[df_post_eda['loan_status']=='Fully Paid']['loan_status'].value_counts(),
                    'text': 'Fully Paid',
                    'name': 'Fully Paid',
                    'type': 'bar'
                }
            ],
            'layout': {}
        }
    ),
    html.H3('Distribution of Loan Amount'),
    dcc.Graph(
        id='loan_amount_dist',
        figure={
            'data': [
                {
                    'x': df_post_eda['loan_amnt'],
                    'text': 'No. of Loans',
                    'type': 'histogram'
                }
            ],
            'layout': {}
        }
    ),
    html.H3('Loan Amount Distribution by Status (Risky Loans Breakdown)'),
    dcc.Graph(
        id='loan_amount_by_status',
        figure={
            'data': [
                {
                    'x': df_post_eda[df_post_eda['loan_status'] == 'Late (31-120 days)']['loan_amnt'].T,
                    'text': 'No. of Late Loans (31-120 days)',
                    'name': 'No. of Late Loans (31-120 days)',
                    'type': 'histogram'
                },
                {
                    'x': df_post_eda[df_post_eda['loan_status'] == 'Charged Off']['loan_amnt'].T,
                    'text': 'No. of Charged Off Loans',
                    'name': 'No. of Charged Off Loans',
                    'type': 'histogram'
                },
                {
                    'x': df_post_eda[df_post_eda['loan_status'] == 'Default']['loan_amnt'].T,
                    'text': 'No. of Default Loans',
                    'name': 'No. of Default Loans',
                    'type': 'histogram'
                },
            ],
            'layout': {}
        }
    ),
    html.H3('Loan Amount Distribution by Grade'),
    dcc.Graph(
        id='loan_amount_by_grade',
        figure = go.Figure(data=data, layout=layout)
    )
])

#################################################### MODELING ####################################################
####################################################          ####################################################

# Data for Modeling Page
df_test = pd.read_csv('df_test_m11.csv', index_col=0)
df_test_pd = pd.read_csv('df_test_m12.csv', index_col=0)

# Independent Variables
x_test_m11 = df_test[features_numerical]
x_test_m12 = df_test_pd
x_test_m13 = df_test['title']
x_test_m14 = df_test['desc']
x_test_m15 = df_test['emp_title']

# Target Variable
y_test = df_test["loan_status"]

sampled_records = [28384, 242322, 94760]
#sampled_records.extend(df_test[df_test['loan_status']=='Fully Paid'].sample(1).index.values)
#sampled_records.extend(df_test[df_test['loan_status']=='Late (31-120 days)'].sample(1).index.values)
#sampled_records.extend(df_test[df_test['loan_status']=='Charged Off'].sample(1).index.values)
#sampled_records.extend(df_test[df_test['loan_status']=='Default'].sample(1).index.values)

x_test_m11 = x_test_m11[x_test_m11.index.isin(sampled_records)]
x_test_m12 = x_test_m12[x_test_m12.index.isin(sampled_records)]
x_test_m13 = x_test_m13[x_test_m13.index.isin(sampled_records)]
x_test_m14 = x_test_m14[x_test_m14.index.isin(sampled_records)]
x_test_m15 = x_test_m15[x_test_m15.index.isin(sampled_records)]

y_test = y_test[y_test.index.isin(sampled_records)]
y_test = np.where(y_test == 'Fully Paid', 0, 1)

rfcm11 = pickle.load(open('rfcm11.sav', 'rb'))
rfcm12 = pickle.load(open('rfcm12.sav', 'rb'))
rfcm13 = pickle.load(open('rfcm13.sav', 'rb'))
rfcm14 = pickle.load(open('rfcm14.sav', 'rb'))
rfcm15 = pickle.load(open('rfcm15.sav', 'rb'))

tv13 = pickle.load(open('tv13.sav', 'rb'))
tv14 = pickle.load(open('tv14.sav', 'rb'))
tv15 = pickle.load(open('tv15.sav', 'rb'))

x_test_m13 = tv13.transform(x_test_m13)
x_test_m14 = tv14.transform(x_test_m14)
x_test_m15 = tv15.transform(x_test_m15)

y_pred_m11 = rfcm11.predict_proba(x_test_m11)
y_pred_m12 = rfcm12.predict_proba(x_test_m12)
y_pred_m13 = rfcm13.predict_proba(x_test_m13)
y_pred_m14 = rfcm14.predict_proba(x_test_m14)
y_pred_m15 = rfcm15.predict_proba(x_test_m15)

df_proba_m1 = pd.DataFrame()
df_proba_m1['y_test'] = y_test
df_proba_m1['y_proba1'] = y_pred_m11[:,1]
df_proba_m1['y_proba2'] = y_pred_m12[:,1]
df_proba_m1['y_proba3'] = y_pred_m13[:,1]
df_proba_m1['y_proba4'] = y_pred_m14[:,1]
df_proba_m1['y_proba5'] = y_pred_m15[:,1]

df_proba_m1['y_proba_conft'] = df_proba_m1['y_proba1']*df_proba_m1['y_proba2']*df_proba_m1['y_proba3']*df_proba_m1['y_proba4']*df_proba_m1['y_proba5']
df_proba_m1['y_proba_confb'] = (1-df_proba_m1['y_proba1'])*(1-df_proba_m1['y_proba2'])*(1-df_proba_m1['y_proba3'])*(1-df_proba_m1['y_proba4'])*(1-df_proba_m1['y_proba5'])
df_proba_m1['y_proba_confa'] = df_proba_m1['y_proba_conft'] / (df_proba_m1['y_proba_conft'] + df_proba_m1['y_proba_confb'])

# Layout for the Modeling Page
base_chart = {
    "values": [40, 10, 10, 10, 10, 10, 10],
    "labels": ["-", "0", "20", "40", "60", "80", "100"],
    "domain": {"x": [0, .48]},
    "marker": {
        "colors": [
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)'
        ],
        "line": {
            "width": 1
        }
    },
    "name": "Gauge",
    "hole": .4,
    "type": "pie",
    "direction": "clockwise",
    "rotation": 108,
    "showlegend": False,
    "hoverinfo": "none",
    "textinfo": "label",
    "textposition": "outside"
}
        
meter_chart1 = {
    "values": [50, 10, 10, 10, 10, 10],
    "labels": ["Risk Score: {}".format(np.round(df_proba_m1.loc[0, 'y_proba_confa']*100, 2)), "Monitor", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
    "marker": {
        'colors': [
            'rgb(255, 255, 255)',
            'rgb(232,226,202)',
            'rgb(226,210,172)',
            'rgb(223,189,139)',
            'rgb(223,162,103)',
            'rgb(226,126,64)'
        ]
    },
    "domain": {"x": [0, 0.48]},
    "name": "Gauge",
    "hole": .3,
    "type": "pie",
    "direction": "clockwise",
    "rotation": 90,
    "showlegend": False,
    "textinfo": "label",
    "textposition": "inside",
    "hoverinfo": "none"
}
    
meter_chart2 = {
    "values": [50, 10, 10, 10, 10, 10],
    "labels": ["Risk Score: {}".format(np.round(df_proba_m1.loc[1, 'y_proba_confa']*100, 2)), "Monitor", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
    "marker": {
        'colors': [
            'rgb(255, 255, 255)',
            'rgb(232,226,202)',
            'rgb(226,210,172)',
            'rgb(223,189,139)',
            'rgb(223,162,103)',
            'rgb(226,126,64)'
        ]
    },
    "domain": {"x": [0, 0.48]},
    "name": "Gauge",
    "hole": .3,
    "type": "pie",
    "direction": "clockwise",
    "rotation": 90,
    "showlegend": False,
    "textinfo": "label",
    "textposition": "inside",
    "hoverinfo": "none"
}
    
meter_chart3 = {
    "values": [50, 10, 10, 10, 10, 10],
    "labels": ["Risk Score: {}".format(np.round(df_proba_m1.loc[2, 'y_proba_confa']*100, 2)), "Monitor", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
    "marker": {
        'colors': [
            'rgb(255, 255, 255)',
            'rgb(232,226,202)',
            'rgb(226,210,172)',
            'rgb(223,189,139)',
            'rgb(223,162,103)',
            'rgb(226,126,64)'
        ]
    },
    "domain": {"x": [0, 0.48]},
    "name": "Gauge",
    "hole": .3,
    "type": "pie",
    "direction": "clockwise",
    "rotation": 90,
    "showlegend": False,
    "textinfo": "label",
    "textposition": "inside",
    "hoverinfo": "none"
}

layout = { 'width': 550, 'height': 500 }

# we don't want the boundary now
base_chart['marker']['line']['width'] = 0

layout_page_3 = html.Div([
    html.H3('Modeling Results'),
    html.Div(className='divTable',children=[
            html.Div(className='divTableBody',children=[
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=[]),
                            html.Div(className='divTableCell',children=[
                                    'User Profile: {}'.format(sampled_records[0])]),
                            html.Div(className='divTableCell',children=[
                                    'User Profile: {}'.format(sampled_records[1])]),
                            html.Div(className='divTableCell',children=[
                                    'User Profile: {}'.format(sampled_records[2])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=[]),
                            html.Div(className='divTableCell',children=[
                                    dcc.Graph(id='risk1',figure = go.Figure(data=[base_chart, meter_chart1], layout=layout))]),
                            html.Div(className='divTableCell',children=[
                                    dcc.Graph(id='risk2',figure = go.Figure(data=[base_chart, meter_chart2], layout=layout))]),
                            html.Div(className='divTableCell',children=[
                                    dcc.Graph(id='risk3',figure = go.Figure(data=[base_chart, meter_chart3], layout=layout))]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['Features']),
                            html.Div(className='divTableCell',children=['Loan Amount: {}'.format(df_test.loc[sampled_records[0], 'loan_amnt'])]),
                            html.Div(className='divTableCell',children=['Loan Amount: {}'.format(df_test.loc[sampled_records[1], 'loan_amnt'])]),
                            html.Div(className='divTableCell',children=['Loan Amount: {}'.format(df_test.loc[sampled_records[2], 'loan_amnt'])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['']),
                            html.Div(className='divTableCell',children=['Loan Purpose: {}'.format(df_test.loc[sampled_records[0], 'purpose'])]),
                            html.Div(className='divTableCell',children=['Loan Purpose: {}'.format(df_test.loc[sampled_records[1], 'purpose'])]),
                            html.Div(className='divTableCell',children=['Loan Purpose: {}'.format(df_test.loc[sampled_records[2], 'purpose'])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['']),
                            html.Div(className='divTableCell',children=['Loan Term: {}'.format(df_test.loc[sampled_records[0], 'term'])]),
                            html.Div(className='divTableCell',children=['Loan Term: {}'.format(df_test.loc[sampled_records[1], 'term'])]),
                            html.Div(className='divTableCell',children=['Loan Term: {}'.format(df_test.loc[sampled_records[2], 'term'])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['']),
                            html.Div(className='divTableCell',children=['Loan Grade: {}'.format(df_test.loc[sampled_records[0], 'grade'])]),
                            html.Div(className='divTableCell',children=['Loan Grade: {}'.format(df_test.loc[sampled_records[1], 'grade'])]),
                            html.Div(className='divTableCell',children=['Loan Grade: {}'.format(df_test.loc[sampled_records[2], 'grade'])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['']),
                            html.Div(className='divTableCell',children=['Received Payment: {}'.format(df_test.loc[sampled_records[0], 'total_rec_prncp'])]),
                            html.Div(className='divTableCell',children=['Received Payment: {}'.format(df_test.loc[sampled_records[1], 'total_rec_prncp'])]),
                            html.Div(className='divTableCell',children=['Received Payment: {}'.format(df_test.loc[sampled_records[2], 'total_rec_prncp'])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['']),
                            html.Div(className='divTableCell',children=['Employee Title: {}'.format(df_test.loc[sampled_records[0], 'emp_title'])]),
                            html.Div(className='divTableCell',children=['Employee Title: {}'.format(df_test.loc[sampled_records[1], 'emp_title'])]),
                            html.Div(className='divTableCell',children=['Employee Title: {}'.format(df_test.loc[sampled_records[2], 'emp_title'])]),
                            ]),
                    html.Div(className='divTableRow',children=[
                            html.Div(className='divTableCell',children=['']),
                            html.Div(className='divTableCell',children=['Employee Salary: {}'.format(df_test.loc[sampled_records[0], 'annual_inc'])]),
                            html.Div(className='divTableCell',children=['Employee Salary: {}'.format(df_test.loc[sampled_records[1], 'annual_inc'])]),
                            html.Div(className='divTableCell',children=['Employee Salary: {}'.format(df_test.loc[sampled_records[2], 'annual_inc'])]),
                            ])
                    ])
            ])
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/intro':
        return layout_page_1
    elif pathname == '/eda':
        return layout_page_2
    elif pathname == '/modeling':
         return layout_page_3
    else:
        pathname = '/eda'
        return layout_page_2

# Add bootstrap css
app.css.append_css({"external_url": [
    "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"
]})

# Add style css    
app.css.append_css({"external_url": [
    "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"
]})

if __name__ == '__main__':
    app.run_server(debug=True)