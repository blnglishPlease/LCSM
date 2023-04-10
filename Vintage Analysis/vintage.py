import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output


credit = pd.read_csv('https://raw.githubusercontent.com/blnglishPlease/LCSM/main/CreditCardApproval/credit_record.csv')

def past_due(x, status):
  if x in 'CX':
    return 0
  else:
    return int(int(x) >= status)

pivot_cr = credit.pivot(index='ID', columns='MONTHS_BALANCE', values='STATUS')
group_cr = credit.groupby(by='ID')

# Убираем карты, по которым не было займов, насколько это корректно?
pivot_cr = pivot_cr[group_cr['STATUS'].unique().apply(lambda x: ('X' not in x) | (len(x) != 1))]

pivot_cr['ID'] = pivot_cr.index
pivot_cr['generation'] = group_cr['MONTHS_BALANCE'].min()
pivot_cr['issues_in_gen'] = pivot_cr.groupby(by='generation')['ID'].transform('count')
pivot_cr.reset_index(drop=True, inplace=True)

# Так как есть кредитные карты, по которым записи обрываются, делаем unpivot заполняя пропуски 'X' как отсутствие займов
unpivot_cr = pivot_cr.melt(id_vars=['ID', 'generation', 'issues_in_gen'], var_name='months_balance', value_name='status').fillna('X')
# И оставляем только записи с момента выдачи
credit_gens = unpivot_cr[unpivot_cr.months_balance >= unpivot_cr.generation]

credit_gens['month_of_booking'] = credit_gens['months_balance'] - credit_gens['generation']

# Флаги просрочки к месяцу наблюдения на 1+, 30+, 60+... дней
credit_gens['nl_past_due1'] = credit_gens['status'].apply(lambda x: past_due(x, 0))
credit_gens['nl_past_due30'] = credit_gens['status'].apply(lambda x: past_due(x, 1))
credit_gens['nl_past_due60'] = credit_gens['status'].apply(lambda x: past_due(x, 2))
credit_gens['nl_past_due90'] = credit_gens['status'].apply(lambda x: past_due(x, 3))
credit_gens['nl_past_due120'] = credit_gens['status'].apply(lambda x: past_due(x, 4))
credit_gens['nl_past_due150'] = credit_gens['status'].apply(lambda x: past_due(x, 5))

# Добавляем Gross Loss флаги
nl_dues_cols = ['nl_past_due1', 'nl_past_due30', 'nl_past_due60', 'nl_past_due90', 'nl_past_due120', 'nl_past_due150']
gl_dues_cols = ['gl_past_due1', 'gl_past_due30', 'gl_past_due60', 'gl_past_due90', 'gl_past_due120', 'gl_past_due150']
rr_dues_cols = ['rr_past_due1', 'rr_past_due30', 'rr_past_due60', 'rr_past_due90', 'rr_past_due120', 'rr_past_due150']
credit_gens.sort_values(by=['ID', 'month_of_booking'], inplace=True)
credit_gens[gl_dues_cols] = credit_gens.groupby(by=['ID'])[nl_dues_cols].cummax()

vintage = credit_gens.groupby(by=['generation', 'month_of_booking', 'issues_in_gen'], as_index=False)[nl_dues_cols + gl_dues_cols].sum()
vintage[nl_dues_cols + gl_dues_cols] = vintage[nl_dues_cols + gl_dues_cols].div(vintage['issues_in_gen'], axis=0) # То же самое, что apply(lambda x: x / vintage['issues_in_gen'])

# Recovery Rate
gl = vintage[gl_dues_cols]
gl.columns = rr_dues_cols
nl = vintage[nl_dues_cols]
nl.columns = rr_dues_cols
vintage[rr_dues_cols] = gl.sub(nl, axis=1)

# Делаем из широкой таблицы длинную
vintage_long = pd.wide_to_long(vintage, stubnames=['nl_past_due', 'gl_past_due', 'rr_past_due'], i=['generation', 'month_of_booking'], j='past_due', suffix='\d+').reset_index()
vintage_long = vintage_long[['generation', 'month_of_booking', 'past_due', 'nl_past_due', 'gl_past_due', 'rr_past_due']]
vintage_long.columns = ['generation', 'month_of_booking', 'past_due', 'nl', 'gl', 'rr']
vintage_long = vintage_long.melt(id_vars=['generation', 'month_of_booking', 'past_due'], var_name='vintage_type', value_name='rate')
vintage_long['generation'] = vintage_long['generation'].astype(str)

data = vintage_long

past_due_options = vintage_long['past_due'].unique()
vintage_type_options = vintage_long['vintage_type'].unique()
vintage_type_dict = {'gl': 'Gross Loss', 'nl': 'Net Loss', 'rr': 'Recovery Rate'}
generations = data.generation.astype(int).unique()

past_due_dropdown = dcc.Dropdown(
    id="past_due_dropdown",
    options=[{'label': str(i) + '+', 'value': i} for i in past_due_options],
    value=90,
    clearable=False,
)

vintage_type_dropdown = dcc.Dropdown(
    id="vintage_type_dropdown",
    options=[{'label': vintage_type_dict[i], 'value': i} for i in vintage_type_options],
    value='gl',
    clearable=False,
)

app = Dash(__name__)


app.layout = html.Div([
    html.H2('Vintage analysis')
    , html.Div(
        [html.P("Select credit card delinquency:")
          , past_due_dropdown]
        , style={'width': '20%', 'display': 'inline-block'})
    , html.Div(
        [html.P("Select vintage type:")
          , vintage_type_dropdown]
        , style={'width': '20%',  'display': 'inline-block'})
    , html.Div(
        [html.P("Generations:")
          , dcc.RangeSlider(id="gen_range_slider", min=generations.min(), max=generations.max(), value=[-10, -1], step=1)
          , dcc.Graph(id="rate_vs_month_of_booking")]
        , style={'width': '80%', 'display': 'inline-block'}),])

@app.callback(
    Output('rate_vs_month_of_booking', 'figure'),
    [Input('past_due_dropdown', 'value'),
     Input('vintage_type_dropdown', 'value'), 
     Input("gen_range_slider", "value")]
)
def update_plot(past_due, vintage_type, gen):
    filtered_data = data[(data['past_due'] == past_due) & (data['vintage_type'] == vintage_type)]
    generations = filtered_data['generation'].unique()
    generations_filtered = generations[(generations.astype(int) >= min(gen)) & (generations.astype(int) <= max(gen))]
    

    traces = []
    for generation in generations_filtered:
        filtered_generation_data = filtered_data[filtered_data['generation'] == generation]
        traces.append(go.Scatter(
            x=filtered_generation_data['month_of_booking'],
            y=filtered_generation_data['rate'],
            mode='lines+markers',
            name=generation
        ))
    
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Month of Booking'},
            yaxis={'title': 'Rate'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest',
            height=800
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)