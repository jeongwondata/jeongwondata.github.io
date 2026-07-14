
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import shapiro, normaltest, zscore, skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq

# =============================================================
# DATA LOADING & PREPROCESSING
# =============================================================
DATA_PATH = 'online_shopping.csv'

df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.drop(columns=['Unnamed: 0'], errors='ignore')

col_required = ['CustomerID','Gender','Location','Tenure_Months','Transaction_ID',
                'Quantity','Avg_Price','Delivery_Charges','Coupon_Status',
                'GST','Offline_Spend','Online_Spend']
df = df_raw.dropna(subset=col_required).copy()
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
df['CustomerID']       = df['CustomerID'].astype(int)
df['Transaction_ID']   = df['Transaction_ID'].astype(int)
df['Total_Spend']      = df['Offline_Spend'] + df['Online_Spend']
df['Revenue']          = df['Quantity'] * df['Avg_Price']
df['Month']            = df['Transaction_Date'].dt.month
df['Price_Bin']        = pd.cut(df['Avg_Price'],
                                 bins=[0,25,50,100,200,400],
                                 labels=['$0-25','$25-50','$50-100','$100-200','$200+'])

CAT_COLS = ['Gender','Location','Product_Category','Coupon_Status']
NUM_COLS = ['Tenure_Months','Quantity','Avg_Price','Delivery_Charges',
            'GST','Offline_Spend','Online_Spend','Discount_pct','Total_Spend','Revenue']

PALETTE  = ['#e991b8','#81c784','#f48fb1','#a5d6a7','#f8b4d9','#b2dfdb','#fce4ec','#e8f5e9','#f06292','#66bb6a']
C_F      = '#e991b8'
C_M      = '#81c784'
TEMPLATE = 'plotly_white'

# =============================================================
# HELPER: standard figure layout
# =============================================================
def apply_layout(fig, title='', xlab='', ylab=''):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#c2185b'), x=0.5),
        xaxis_title=xlab, yaxis_title=ylab,
        template=TEMPLATE, margin=dict(t=60, b=50, l=60, r=30),
        legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#ccc', borderwidth=1),
        font=dict(family='Arial', size=13)
    )
    return fig

# =============================================================
# APP INIT
# =============================================================
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.MINTY],
                suppress_callback_exceptions=True)
app.title = 'Online Shopping Dashboard — DATS6401'
server = app.server

# =============================================================
# NAVBAR
# =============================================================
navbar = dbc.NavbarSimple(
    brand='🛒 Online Shopping Analytics Dashboard',
    brand_href='#',
    color='#f48fb1',
    dark=False,
    children=[
        dbc.NavItem(dbc.NavLink('DATS6401 | Jeongwon Yoo', href='#')),
    ]
)

# =============================================================
# TAB CONTENTS
# =============================================================

# ── TAB 1: Load Data ─────────────────────────────────────────
tab_load = dbc.Tab(label='📂 Load Data', tab_id='tab-load', children=[
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Dataset Overview', className='mb-0')),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select display rows:'),
                                dbc.Select(id='load-rows',
                                           options=[{'label':f'{n} rows','value':n}
                                                    for n in [5,10,20,50]],
                                           value=10),
                            ], width=3),
                            dbc.Col([
                                html.Label('Filter by Column:'),
                                dbc.Select(id='load-col-filter',
                                           options=[{'label':'All','value':'All'}] +
                                                   [{'label':c,'value':c} for c in df.columns],
                                           value='All'),
                            ], width=3),
                            dbc.Col([
                                dbc.Button('Load Data', id='btn-load', color='success',
                                           className='mt-4'),
                                dbc.Spinner(html.Div(id='load-status'), size='sm'),
                            ], width=3),
                        ], className='mb-3'),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H6('Total Rows'),
                                    dbc.Badge(f'{len(df):,}', color='success', className='fs-5'),
                                ])
                            ]), width=2),
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H6('Columns'),
                                    dbc.Badge(f'{df.shape[1]}', color='success', className='fs-5'),
                                ])
                            ]), width=2),
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H6('Missing Values'),
                                    dbc.Badge(f'{df.isnull().sum().sum()}', color='success', className='fs-5'),
                                ])
                            ]), width=2),
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H6('Unique Customers'),
                                    dbc.Badge(f'{df["CustomerID"].nunique():,}', color='danger', className='fs-5'),
                                ])
                            ]), width=2),
                            dbc.Col(dbc.Card([
                                dbc.CardBody([
                                    html.H6('Date Range'),
                                    dbc.Badge(f'{df["Transaction_Date"].min().date()} ~ {df["Transaction_Date"].max().date()}',
                                              color='secondary', className='fs-6'),
                                ])
                            ]), width=4),
                        ], className='mb-3'),
                        html.Div(id='load-table-div'),
                    ])
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Descriptive Statistics')),
                    dbc.CardBody([
                        html.Div(id='load-stats-div'),
                    ])
                ])
            ])
        ], className='mt-3'),
    ], fluid=True, className='py-3')
])

# ── TAB 2: Data Cleaning ─────────────────────────────────────
tab_clean = dbc.Tab(label='🧹 Data Cleaning', tab_id='tab-clean', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Data Cleaning Methods')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('Cleaning Method:'),
                        dbc.RadioItems(
                            id='clean-method',
                            options=[
                                {'label': 'Drop NA rows', 'value': 'drop'},
                                {'label': 'Fill with Mean', 'value': 'mean'},
                                {'label': 'Fill with Median', 'value': 'median'},
                                {'label': 'Fill with Mode', 'value': 'mode'},
                                {'label': 'Forward Fill', 'value': 'ffill'},
                            ],
                            value='drop', inline=False,
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label('Select Feature to Check:'),
                        dbc.Select(id='clean-feature',
                                   options=[{'label': c, 'value': c} for c in
                                            ['Tenure_Months', 'Quantity', 'Avg_Price',
                                             'Delivery_Charges', 'GST',
                                             'Offline_Spend', 'Online_Spend', 'Discount_pct']],
                                   value='Online_Spend'),
                        html.Br(),
                        dbc.Button('Apply Cleaning', id='btn-clean', color='success'),
                    ], width=3),
                    dbc.Col([
                        dbc.Alert(id='clean-alert', color='info',
                                  children='Select a method and click Apply.'),
                        dbc.Row([
                            dbc.Col(daq.LEDDisplay(id='led-missing-before',
                                                   label='Missing Before',
                                                   value='0', size=24,
                                                   color='#e991b8'), width=6),
                            dbc.Col(daq.LEDDisplay(id='led-missing-after',
                                                   label='Missing After',
                                                   value='0', size=24,
                                                   color='#81c784'), width=6),
                        ]),
                    ], width=6),
                ], className='mb-3'),
                dcc.Graph(id='clean-graph'),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 3: Outlier Detection ─────────────────────────────────
tab_outlier = dbc.Tab(label='🔍 Outlier Detection', tab_id='tab-outlier', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Outlier Detection & Removal')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('Detection Method:'),
                        dbc.RadioItems(
                            id='outlier-method',
                            options=[
                                {'label': 'IQR (1.5×)', 'value': 'iqr'},
                                {'label': 'Z-Score (|z|>3)', 'value': 'zscore'},
                                {'label': 'Modified Z-Score', 'value': 'mod_z'},
                            ],
                            value='iqr',
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label('Feature:'),
                        dbc.Select(id='outlier-feature',
                                   options=[{'label':c,'value':c} for c in NUM_COLS],
                                   value='Online_Spend'),
                        html.Br(),
                        html.Label('IQR Multiplier:'),
                        dcc.Slider(id='outlier-k', min=1.0, max=3.0, step=0.5,
                                   value=1.5, marks={1.0:'1.0',1.5:'1.5',2.0:'2.0',
                                                     2.5:'2.5',3.0:'3.0'}),
                        dbc.Button('Detect Outliers', id='btn-outlier', color='danger', className='mt-2'),
                    ], width=4),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(daq.Gauge(id='gauge-outlier-pct',
                                             label='Outlier %',
                                             min=0, max=30, value=0,
                                             color={'gradient':True,
                                                    'ranges':{'green':[0,5],
                                                              'yellow':[5,15],
                                                              'red':[15,30]}},
                                             size=150), width=6),
                            dbc.Col([
                                daq.LEDDisplay(id='led-outlier-n', label='# Outliers',
                                               value='0', size=24, color='#e991b8'),
                                html.Br(),
                                daq.LEDDisplay(id='led-clean-n', label='Cleaned Rows',
                                               value=str(len(df)), size=24, color='#81c784'),
                            ], width=6),
                        ]),
                    ], width=5),
                ], className='mb-3'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='outlier-box'), width=6),
                    dbc.Col(dcc.Graph(id='outlier-hist'), width=6),
                ]),
                dbc.Alert(id='outlier-alert', color='info',
                          children='Select method and click Detect.'),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 4: PCA ───────────────────────────────────────────────
tab_pca = dbc.Tab(label='📉 PCA', tab_id='tab-pca', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Principal Component Analysis (PCA)')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('Features to include:'),
                        dbc.Checklist(
                            id='pca-features',
                            options=[{'label':c,'value':c} for c in NUM_COLS],
                            value=NUM_COLS,
                            inline=False,
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label('Number of Components:'),
                        dcc.Slider(id='pca-n', min=2, max=10, step=1, value=5,
                                   marks={i:str(i) for i in range(2,11)}),
                        html.Br(),
                        html.Label('Color by:'),
                        dbc.Select(id='pca-color',
                                   options=[{'label':c,'value':c} for c in CAT_COLS],
                                   value='Gender'),
                        html.Br(),
                        dbc.Button('Run PCA', id='btn-pca', color='success'),
                    ], width=4),
                    dbc.Col([
                        daq.GraduatedBar(id='grad-bar-pca',
                                         label='Cumulative Variance (%)',
                                         min=0, max=100, value=0,
                                         showCurrentValue=True, size=200,
                                         color={'gradient':True,
                                                'ranges':{'red':[0,60],
                                                          'yellow':[60,80],
                                                          'green':[80,100]}}),
                        html.Br(),
                        dbc.Alert(id='pca-cond-alert', color='secondary',
                                  children='Condition number will appear here.'),
                    ], width=5),
                ], className='mb-3'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='pca-scree'), width=6),
                    dbc.Col(dcc.Graph(id='pca-biplot'), width=6),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='pca-cumvar'), width=12),
                ]),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 5: Normality Test ────────────────────────────────────
tab_normality = dbc.Tab(label='📊 Normality Test', tab_id='tab-norm', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Normality Tests')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('Feature:'),
                        dbc.Select(id='norm-feature',
                                   options=[{'label':c,'value':c} for c in NUM_COLS],
                                   value='Online_Spend'),
                        html.Br(),
                        html.Label('Test Method:'),
                        dbc.RadioItems(
                            id='norm-method',
                            options=[
                                {'label': "Shapiro-Wilk", 'value': 'shapiro'},
                                {'label': "D'Agostino K²", 'value': 'dagostino'},
                                {'label': 'Both', 'value': 'both'},
                            ],
                            value='both',
                        ),
                        html.Br(),
                        dbc.Button('Run Test', id='btn-norm', color='success'),
                    ], width=3),
                    dbc.Col([
                        dbc.Alert(id='norm-result', color='secondary',
                                  children='Select feature and run test.'),
                        dbc.Row([
                            dbc.Col(daq.Thermometer(id='thermo-skew',
                                                    label='Skewness',
                                                    min=-5, max=5, value=0,
                                                    showCurrentValue=True,
                                                    height=120, width=20,
                                                    color='#81c784'), width=4),
                            dbc.Col(daq.Thermometer(id='thermo-kurt',
                                                    label='Kurtosis',
                                                    min=-3, max=10, value=0,
                                                    showCurrentValue=True,
                                                    height=120, width=20,
                                                    color='#e991b8'), width=4),
                            dbc.Col([
                                daq.BooleanSwitch(id='norm-switch',
                                                  label='Normal?',
                                                  on=False,
                                                  color='#81c784'),
                            ], width=4),
                        ]),
                    ], width=4),
                    dbc.Col([
                        daq.Knob(id='knob-pval',
                                 label='p-value',
                                 min=0, max=1, value=0,
                                 size=120,
                                 color={'gradient':True,
                                        'ranges':{'#81c784':[0,0.05],
                                                  'yellow':[0.05,0.1],
                                                  '#e991b8':[0.1,1]}}),

                    ], width=2),
                    dbc.Col([
                        daq.ColorPicker(id='norm-color',
                                        label='Plot Color',
                                        value=dict(hex='#e991b8')),
                    ], width=3),
                ], className='mb-3'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='norm-hist'), width=6),
                    dbc.Col(dcc.Graph(id='norm-qq'), width=6),
                ]),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 6: Data Transformation ───────────────────────────────
tab_transform = dbc.Tab(label='🔄 Data Transformation', tab_id='tab-transform', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Data Transformation')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('Feature:'),
                        dbc.Select(id='trans-feature',
                                   options=[{'label':c,'value':c} for c in NUM_COLS],
                                   value='Online_Spend'),
                        html.Br(),
                        html.Label('Transformation Method:'),
                        dbc.Select(id='trans-method',
                                   options=[
                                       {'label':'Log (log1p)','value':'log'},
                                       {'label':'Square Root','value':'sqrt'},
                                       {'label':'Z-Score Standardization','value':'zscore'},
                                       {'label':'Min-Max Normalization','value':'minmax'},
                                       {'label':'Box-Cox','value':'boxcox'},
                                   ],
                                   value='log'),
                        html.Br(),
                        dbc.Button('Apply', id='btn-transform', color='success'),
                    ], width=3),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.P('Before:', style={'fontWeight':'bold'}),
                                daq.LEDDisplay(id='led-skew-before', label='Skewness',
                                               value='0.00', size=20, color='#e991b8'),
                                html.Br(),
                                daq.LEDDisplay(id='led-mean-before', label='Mean',
                                               value='0.00', size=20, color='#81c784'),
                            ], width=6),
                            dbc.Col([
                                html.P('After:', style={'fontWeight':'bold'}),
                                daq.LEDDisplay(id='led-skew-after', label='Skewness',
                                               value='0.00', size=20, color='#81c784'),
                                html.Br(),
                                daq.LEDDisplay(id='led-mean-after', label='Mean',
                                               value='0.00', size=20, color='#a5d6a7'),
                            ], width=6),
                        ]),
                        dbc.Alert(id='transform-alert', color='info',
                                  children='Select a feature and transformation.'),
                    ], width=4),
                    dbc.Col([
                        dcc.Download(id='download-transformed'),
                        dbc.Button('⬇ Download Transformed CSV',
                                   id='btn-download-transform',
                                   color='danger', outline=True, className='mt-2'),
                    ], width=5),
                ], className='mb-3'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='trans-before'), width=6),
                    dbc.Col(dcc.Graph(id='trans-after'), width=6),
                ]),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 7: Numerical Feature Plots ───────────────────────────
tab_num = dbc.Tab(label='📈 Numerical Plots', tab_id='tab-num', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Numerical Feature Visualization')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('X Feature:'),
                        dbc.Select(id='num-x',
                                   options=[{'label':c,'value':c} for c in NUM_COLS],
                                   value='Online_Spend'),
                        html.Br(),
                        html.Label('Y Feature (for 2D plots):'),
                        dbc.Select(id='num-y',
                                   options=[{'label':c,'value':c} for c in NUM_COLS],
                                   value='Offline_Spend'),
                        html.Br(),
                        html.Label('Color by:'),
                        dbc.Select(id='num-hue',
                                   options=[{'label':'None','value':'None'}] +
                                           [{'label':c,'value':c} for c in CAT_COLS],
                                   value='Gender'),
                    ], width=3),
                    dbc.Col([
                        html.Label('Plot Type:'),
                        dbc.DropdownMenu(
                            label='Select Plot Type',
                            id='num-plot-dropdown',
                            children=[
                                dbc.DropdownMenuItem('Histogram + KDE',  id='dd-hist'),
                                dbc.DropdownMenuItem('KDE Plot',          id='dd-kde'),
                                dbc.DropdownMenuItem('Box Plot',          id='dd-box'),
                                dbc.DropdownMenuItem('Violin Plot',       id='dd-violin'),
                                dbc.DropdownMenuItem('Scatter + Regression', id='dd-scatter'),
                                dbc.DropdownMenuItem('QQ Plot',           id='dd-qq'),
                                dbc.DropdownMenuItem('Dist Plot',         id='dd-dist'),
                                dbc.DropdownMenuItem('Area Plot',         id='dd-area'),
                                dbc.DropdownMenuItem('Rug Plot',          id='dd-rug'),
                                dbc.DropdownMenuItem('Hexbin / Density',  id='dd-hexbin'),
                                dbc.DropdownMenuItem('3D Scatter',        id='dd-scatter3d'),
                                dbc.DropdownMenuItem('Contour',           id='dd-contour'),
                                dbc.DropdownMenuItem('Joint Plot',        id='dd-joint'),
                            ],
                            color='secondary', className='mb-2',
                        ),
                        dbc.Tooltip('Choose a plot type from the dropdown menu.',
                                    target='num-plot-dropdown'),
                        dcc.Store(id='num-plot-type', data='hist'),
                        html.Br(),
                        html.Label('Bin Range (for histogram):'),
                        dcc.RangeSlider(id='num-bins', min=10, max=100, step=10,
                                        value=[10, 50],
                                        marks={10:'10', 30:'30', 50:'50',
                                               70:'70', 100:'100'}),
                        dbc.Tooltip('Drag both handles to set the min and max bin count.',
                                    target='num-bins'),
                        html.Br(),
                        dbc.Button('Plot', id='btn-num-plot', color='success'),
                        dbc.Tooltip('Click to generate the selected plot.',
                                    target='btn-num-plot'),
                    ], width=4),
                    dbc.Col([
                        dbc.Alert(id='num-stats-alert', color='light',
                                  children='Stats will appear here.'),
                        html.Label('Observations / Notes:'),
                        dcc.Textarea(id='num-textarea',
                                     placeholder='Write your observations about this plot here...',
                                     style={'width':'100%', 'height':'120px',
                                            'fontSize':'13px'}),
                        dbc.Tooltip('Use this area to note your observations about the plot.',
                                    target='num-textarea'),
                    ], width=5),
                ], className='mb-3'),
                dbc.Spinner(dcc.Graph(id='num-graph', style={'height':'550px'})),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 8: Categorical Feature Plots ─────────────────────────
tab_cat = dbc.Tab(label='📊 Categorical Plots', tab_id='tab-cat', children=[
    dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H5('Categorical Feature Visualization')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('Categorical Feature:'),
                        dbc.Select(id='cat-x',
                                   options=[{'label':c,'value':c} for c in CAT_COLS],
                                   value='Product_Category'),
                        html.Br(),
                        html.Label('Value Feature (Y-axis):'),
                        dbc.Select(id='cat-y',
                                   options=[{'label':c,'value':c} for c in NUM_COLS],
                                   value='Revenue'),
                        html.Br(),
                        html.Label('Hue (Group by):'),
                        dbc.Select(id='cat-hue',
                                   options=[{'label':'None','value':'None'}] +
                                           [{'label':c,'value':c} for c in CAT_COLS],
                                   value='Gender'),
                    ], width=3),
                    dbc.Col([
                        html.Label('Plot Type:'),
                        dbc.Select(id='cat-plot-type',
                                   options=[
                                       {'label':'Bar Plot (sorted)','value':'bar'},
                                       {'label':'Stacked Bar','value':'bar_stack'},
                                       {'label':'Grouped Bar','value':'bar_group'},
                                       {'label':'Count Plot','value':'count'},
                                       {'label':'Pie Chart','value':'pie'},
                                       {'label':'Box Plot','value':'box'},
                                       {'label':'Violin Plot','value':'violin'},
                                       {'label':'Strip Plot','value':'strip'},
                                       {'label':'Swarm Plot','value':'swarm'},
                                       {'label':'Boxen Plot','value':'boxen'},
                                   ],
                                   value='bar'),
                        html.Br(),
                        html.Label('Aggregation (for bar):'),
                        dbc.RadioItems(id='cat-agg',
                                       options=[
                                           {'label':'Mean','value':'mean'},
                                           {'label':'Sum','value':'sum'},
                                           {'label':'Count','value':'count'},
                                           {'label':'Median','value':'median'},
                                       ],
                                       value='mean', inline=True),
                        html.Br(),
                        dbc.Button('Plot', id='btn-cat-plot', color='success'),
                    ], width=4),
                    dbc.Col([
                        dbc.Alert(id='cat-info-alert', color='light',
                                  children='Category info will appear here.'),
                    ], width=5),
                ], className='mb-3'),
                dbc.Spinner(dcc.Graph(id='cat-graph', style={'height':'550px'})),
            ])
        ]),
    ], fluid=True, className='py-3')
])

# ── TAB 9: Statistics ────────────────────────────────────────
tab_stats = dbc.Tab(label='📐 Statistics', tab_id='tab-stats', children=[
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Heatmap & Correlation')),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('Features:'),
                                dbc.Checklist(
                                    id='stats-heatmap-features',
                                    options=[{'label':c,'value':c} for c in NUM_COLS],
                                    value=NUM_COLS[:6],
                                    inline=False,
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label('Color Scale:'),
                                dbc.Select(id='stats-cmap',
                                           options=[
                                               {'label':'RdBu','value':'RdBu'},
                                               {'label':'Viridis','value':'Viridis'},
                                               {'label':'Plasma','value':'Plasma'},
                                               {'label':'Coolwarm','value':'RdYlBu'},
                                           ],
                                           value='RdBu'),
                                html.Br(),
                                dbc.Button('Update', id='btn-stats-heat', color='success'),
                            ], width=3),
                            dbc.Col([
                                dbc.Alert(id='stats-corr-alert', color='light',
                                          children='Highest correlation will appear here.'),
                            ], width=6),
                        ], className='mb-2'),
                        dcc.Graph(id='stats-heatmap'),
                    ])
                ]),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Scatter Matrix (Pair Plot)')),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select features (max 5):'),
                                dbc.Checklist(
                                    id='stats-pair-features',
                                    options=[{'label':c,'value':c} for c in NUM_COLS],
                                    value=['Avg_Price','Online_Spend',
                                           'Offline_Spend','Revenue'],
                                    inline=False,
                                ),
                            ], width=5),
                            dbc.Col([
                                html.Label('Color by:'),
                                dbc.Select(id='stats-pair-hue',
                                           options=[{'label':c,'value':c}
                                                    for c in CAT_COLS],
                                           value='Gender'),
                                html.Br(),
                                dbc.Button('Update', id='btn-stats-pair', color='success'),
                            ], width=7),
                        ], className='mb-2'),
                        dbc.Spinner(dcc.Graph(id='stats-pair')),
                    ])
                ]),
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Multivariate KDE & Statistics Table')),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('X Feature:'),
                                dbc.Select(id='stats-kde-x',
                                           options=[{'label':c,'value':c}
                                                    for c in NUM_COLS],
                                           value='Online_Spend'),
                                html.Label('Y Feature:'),
                                dbc.Select(id='stats-kde-y',
                                           options=[{'label':c,'value':c}
                                                    for c in NUM_COLS],
                                           value='Offline_Spend'),
                                html.Br(),
                                dbc.Button('Update KDE', id='btn-stats-kde',
                                           color='success'),
                            ], width=3),
                            dbc.Col(dcc.Graph(id='stats-kde'), width=9),
                        ]),
                    ])
                ]),
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Descriptive Statistics Table')),
                    dbc.CardBody([
                        html.Div(id='stats-table-div'),
                    ])
                ]),
            ], width=5),
        ]),
    ], fluid=True, className='py-3')
])

# =============================================================
# MAIN LAYOUT
# =============================================================
app.layout = dbc.Container([
    navbar,
    html.Br(),
    dbc.Tabs([
        tab_load,
        tab_clean,
        tab_outlier,
        tab_pca,
        tab_normality,
        tab_transform,
        tab_num,
        tab_cat,
        tab_stats,
    ], id='main-tabs', active_tab='tab-load'),
], fluid=True)
