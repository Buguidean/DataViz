import altair as alt
import pandas as pd
import numpy as np

# Load precomputed data from CSVs
region_year = pd.read_csv('region_year.csv')
metrics_long = pd.read_csv('metrics_long.csv')
country_year = pd.read_csv('country_year.csv')

# Q2 data
steps = pd.read_csv('steps.csv')
vfc_start = pd.read_csv('vfc_start.csv')
delta_df = pd.read_csv('delta_df.csv')

# Q3 data
covid_vfc = pd.read_csv('covid_vfc.csv')
cv_max_delta_df = pd.read_csv('cv_max_delta_df.csv')

# Q4 data
complete_us_access = pd.read_csv('complete_us_access.csv')
f_long_us_access = pd.read_csv('f_long_us_access.csv')
hi_df = pd.read_csv('hi_df.csv')

# Controls
min_year = int(country_year['year'].min())
max_year = int(country_year['year'].max())

year_sel = alt.param(
    name='Year',
    value=max_year,
    bind=alt.binding_range(min=min_year, max=max_year, step=1, name='Year')
)

trend_metric_sel = alt.param(
    name='trend_metric',
    value='Unweighted mean',
    bind=alt.binding_radio(options=['Unweighted mean', 'Population-weighted mean'], name='Trend metric (Q1): ')
)

map_metric_sel = alt.param(
    name='map_metric',
    value='Visa-free count',
    bind=alt.binding_radio(options=['Visa-free count', 'Weighted contribution share'], name='Map metric (Q1): ')
)

region_pick = alt.selection_point(fields=['region'], on='click', clear='dblclick', empty='all')

# ------------------------
# Left column: Bars (top) + Trend (bottom)
# ------------------------

REGION_DOMAIN = [
    "EUROPE", "AMERICAS", "CARIBBEAN", "OCEANIA",
    "ASIA", "MIDDLE EAST", "AFRICA"
]
REGION_RANGE = [
    "#efb118",
    "#3ca951",
    "#ff8ab7",
    "#9c6b4e",
    "#a463f2",
    "#f58518",
    "#6cc5b0",
]

bar_base = alt.Chart(region_year).transform_filter(alt.datum.year == year_sel)

unweighted = (
    bar_base
    .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
    .encode(
        y=alt.Y('region:N', title='', sort='-x'),
        x=alt.X('mean_vfc_unweighted:Q', title='',
                scale=alt.Scale(domain=[0, 200])),
        color=alt.Color(
            'region:N',
            scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE),
            legend=None
        ),
    )
    .properties(height=140, width=340, title='Mean visa-free destinations')
)

weighted = (
    bar_base
    .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
    .encode(
        y=alt.Y('region:N', title='', sort=alt.SortField(field='mean_vfc_weighted', order='descending')),
        x=alt.X('mean_vfc_weighted:Q', title='',
                scale=alt.Scale(domain=[0, 200])),
        color=alt.Color(
            'region:N',
            scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE),
            legend=None
        ),
    )
    .properties(height=140, width=340, title='Population-weighted mean visa-free destinations')
)

bars_col = alt.vconcat(unweighted, weighted, spacing=16)

trend = (
    alt.Chart(metrics_long)
    .transform_filter('datum.metric == trend_metric')
    .mark_line(point=True)
    .encode(
        x=alt.X('year:O', title='', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('value:Q', title=''),
        color=alt.Color(
            'region:N',
            title='Region',
            scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE)
        ),
    )
    .properties(title='Trend by region of mean visa-free destinations', height=180, width=340)
)

left_col = alt.vconcat(bars_col, trend, spacing=16)

# ---------------------------------------
# Right column: Map
# ---------------------------------------

VFC_DOMAIN = [0, 200]
CONTRIB_DOMAIN = [0, 0.7]

VFC_SCHEME = 'cividis'
CONTRIB_SCHEME = 'inferno'

world = alt.topo_feature(
    'https://raw.githubusercontent.com/micahstubbs/tiny-countries-geo/5d591b1d71a6e27e9385d0f2d28855a19314dd7b/out/ne_10m_admin_0_countries-1-percent.topojson',
    'ne_10m_admin_0_countries'
)

map_base = alt.Chart(world).transform_calculate(
    lookup_key='datum.properties.ADM0_A3 + "_" + toString(Year)'
)

map_vfc = (
    map_base
    .mark_geoshape()
    .transform_lookup(
        lookup='lookup_key',
        from_=alt.LookupData(
            country_year,
            key='lookup_key',
            fields=['vfc', 'country', 'code3', 'region']
        )
    )
    .encode(
        color=alt.Color(
            'vfc:Q',
            title='Visa-free count',
            scale=alt.Scale(domain=VFC_DOMAIN, scheme=VFC_SCHEME),
            legend=alt.Legend(format='d')
        ),
        opacity=alt.condition(region_pick, alt.value(1.0), alt.value(0.25)),
        stroke=alt.condition(region_pick, alt.value('black'), alt.value('white')),
        strokeWidth=alt.condition(region_pick, alt.value(0.6), alt.value(0.4)),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('vfc:Q', title='Visa-free count', format='.0f')
        ]
    )
    .transform_filter('map_metric == "Visa-free count"')
)

map_contrib = (
    map_base
    .mark_geoshape()
    .transform_lookup(
        lookup='lookup_key',
        from_=alt.LookupData(
            country_year,
            key='lookup_key',
            fields=['contrib', 'vfc', 'pop', 'country', 'region']
        )
    )
    .encode(
        color=alt.Color(
            'contrib:Q',
            title=['Share of weighted', 'visa-free'],
            scale=alt.Scale(domain=CONTRIB_DOMAIN, scheme=CONTRIB_SCHEME),
            legend=alt.Legend(format='.0%')
        ),
        opacity=alt.condition(region_pick, alt.value(1.0), alt.value(0.25)),
        stroke=alt.condition(region_pick, alt.value('black'), alt.value('white')),
        strokeWidth=alt.condition(region_pick, alt.value(0.6), alt.value(0.4)),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('vfc:Q', title='Visa-free count', format='.0f'),
            alt.Tooltip('pop:Q', title='Population', format='.3s'),
            alt.Tooltip('contrib:Q', title='Share of region total', format='.1%')
        ]
    )
    .transform_filter('map_metric == "Weighted contribution share"')
)

map_chart = (
    (map_vfc + map_contrib)
    .resolve_scale(color='independent')
    .project('equalEarth')
    .properties(
        title='',
        width=720,
        height=420
    )
    .add_params(region_pick)
)

right_col = map_chart

# ------------------------
# Final layout (Q1)
# ------------------------
q1_final = alt.hconcat(
    left_col, right_col, spacing=16
).add_params(
    year_sel, trend_metric_sel, map_metric_sel
).properties(
    title={
        "text": "Q1 - Which region has the most visa-free destinations?",
        "subtitle": "Disclaimer: visa-free count for years 2007 and 2009 has been interpolated."
    }
)

# ------------------------
# Parameters
# ------------------------
start_year = 2006
end_year = 2021
years_order = list(range(start_year, end_year + 1))

delta_metric_sel = alt.param(
    name='delta_metric',
    value='YoY',
    bind=alt.binding_radio(options=['YoY', 'Overall'], name='Delta metric: ')
)

# Controls (with default values)
country_pick = alt.selection_point(
    fields=['country'],
    on='click',
    clear='dblclick',
    empty='none',
    value='Spain',
)
region_pick = alt.selection_point(
    fields=['region'],
    on='click',
    clear='dblclick',
    empty='all',
    value='EUROPE',
)

# ------------------------
# Waterfall (left)
# ------------------------

bars = (
    alt.Chart(steps, width=340, height=180)
    .transform_filter(country_pick)
    .mark_bar()
    .encode(
        x=alt.X('year:O', title='', sort=years_order, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('y0:Q', title='Visa-free destinations', scale=alt.Scale(zero=True)),
        y2='y1:Q',
        color=alt.Color(
            'sign:N',
            title='Change',
            scale=alt.Scale(
                domain=['Decrease', 'Increase', 'No change'],
                range=['#df745e', '#2f78b3', '#bdbdbd']
            )
        ),
        tooltip=[
            alt.Tooltip('y0:Q', title='Level before', format='.0f'),
            alt.Tooltip('y1:Q', title='Level after', format='.0f'),
            alt.Tooltip('delta:Q', title='YoY change', format='+.0f')
        ]
    )
)

start_point = (
    alt.Chart(vfc_start, width=340, height=180)
    .transform_filter(country_pick)
    .mark_bar(filled=True, color='#888')
    .encode(
        x=alt.X('year:O', sort=years_order, title='', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('visa_free_count:Q', title='Visa-free destinations'),
        tooltip=[
            alt.Tooltip('visa_free_count:Q', title='Level', format='.0f')
        ]
    )
)

waterfall = (bars + start_point)

country_label = (
    alt.Chart(vfc_start)
    .transform_filter(country_pick)
    .transform_aggregate(country='max(country)')
    .transform_calculate(
        label="datum.country + ' - Visa-free access changes'"
    )
    .mark_text(align='center', fontSize=14, fontWeight='bold', dy=0)
    .encode(text=alt.Text('label:N'))
    .properties(width=340)
)

first_left_col = alt.vconcat(
    country_label,
    waterfall,
    spacing=0
).resolve_scale(color='independent')

# ------------------------
# Delta bars (left)
# ------------------------

bar_top = (
    alt.Chart(delta_df)
    .transform_filter(region_pick)
    .transform_calculate(
        chosen_delta = "delta_metric == 'YoY' ? datum.max_delta : datum.ov_delta",
        chosen_abs   = "delta_metric == 'YoY' ? datum.max_abs_delta : datum.ov_abs_delta"
    )
    .transform_window(
        rn='row_number()',
        sort=[
            {'field': 'chosen_abs', 'order': 'descending'},
            {'field': 'chosen_delta', 'order': 'descending'},
            {'field': 'country', 'order': 'ascending'}
        ]
    )
    .transform_filter(alt.datum.rn <= 10)
    .mark_bar()
    .encode(
        x=alt.X(
            'code3:N',
            title='',
            sort=alt.SortField(field='rn', order='ascending'),
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y('chosen_delta:Q', title='Delta'),
        color=alt.Color(
          'region:N',
          legend=alt.Legend(title='Region'),
          scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE)
        ),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('max_delta:Q', title='Max YoY delta', format='+.0f'),
            alt.Tooltip('ov_delta:Q', title='Overall delta', format='+.0f'),
        ]
    )
    .properties(
        width=340,
        height=160,
        title='Top 10 countries by change in visa-free destinations'
    )
    .resolve_scale(color='independent')
)

left_col = alt.vconcat(
    first_left_col,
    bar_top,
    spacing=16
).resolve_scale(color='independent')

# ------------------------
# Map (right)
# ------------------------

map_base = alt.Chart(world).transform_calculate(
    lookup_key='datum.properties.ADM0_A3'
)

min_yoy = float(np.floor(delta_df['max_delta'].min()))
max_yoy = float(np.ceil(delta_df['max_delta'].max()))
red_steps_yoy = np.linspace(min_yoy, 0, 3)
mid_red_yoy = float(red_steps_yoy[1])
blue_steps_yoy = list(map(float, np.linspace(0, max_yoy, 4 + 1)[1:]))

DOMAIN_YOY = [min_yoy, mid_red_yoy, 0.0] + blue_steps_yoy

min_ov = float(np.floor(delta_df['ov_delta'].min()))
max_ov = float(np.ceil(delta_df['ov_delta'].max()))
red_steps_ov = np.linspace(min_ov, 0, 3)
mid_red_ov = float(red_steps_ov[1])
blue_steps_ov = list(map(float, np.linspace(0, max_ov, 4 + 1)[1:]))

DOMAIN_OV = [min_ov, mid_red_ov, 0.0] + blue_steps_ov

DIVERGENT_RANGE = [
    "#df745e",  # deeper red
    "#fbdbc9",  # lighter red
    "#f2f3f4",  # white at zero
    "#d2e5ef",  # light blue
    "#9dcae1",  # medium-light blue
    "#5da2cb",  # medium-dark blue
    "#2f78b3"   # dark blue
]

map_max_yoy = (
    map_base
    .mark_geoshape()
    .transform_lookup(
        lookup='lookup_key',
        from_=alt.LookupData(
            delta_df,
            key='code3',
            fields=['max_delta', 'country', 'code3', 'region']
        )
    )
    .encode(
        color=alt.Color(
            'max_delta:Q',
            title='Max YoY delta',
            scale=alt.Scale(domain=DOMAIN_YOY, range=DIVERGENT_RANGE),
            legend=alt.Legend(format='d')
        ),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.6),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('max_delta:Q', title='Max YoY delta', format='.0f')
        ]
    )
    .transform_filter("delta_metric == 'YoY'")
)

map_overall = (
    map_base
    .mark_geoshape()
    .transform_lookup(
        lookup='lookup_key',
        from_=alt.LookupData(
            delta_df,
            key='code3',
            fields=['ov_delta', 'country', 'code3', 'region']
        )
    )
    .encode(
        color=alt.Color(
            'ov_delta:Q',
            title='Overall delta',
            scale=alt.Scale(domain=DOMAIN_OV, range=DIVERGENT_RANGE),
            legend=alt.Legend(format='d')
        ),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.6),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('ov_delta:Q', title='Overall delta', format='.0f')
        ]
    )
    .transform_filter("delta_metric == 'Overall'")
)

# Highlight layers
map_highlight_halo = (
    map_base
    .mark_geoshape(filled=False, stroke='white', strokeWidth=2.4)
    .transform_lookup(
        lookup='lookup_key',
        from_=alt.LookupData(
            delta_df,
            key='code3',
            fields=['country', 'code3', 'region']
        )
    )
    .transform_filter(region_pick)
    .transform_filter(country_pick)
)
map_highlight_outline = (
    map_base
    .mark_geoshape(filled=False, stroke='black', strokeWidth=1.7)
    .transform_lookup(
        lookup='lookup_key',
        from_=alt.LookupData(
            delta_df,
            key='code3',
            fields=['country', 'code3', 'region']
        )
    )
    .transform_filter(region_pick)
    .transform_filter(country_pick)
)

map_chart = (
    (map_max_yoy + map_overall + map_highlight_halo + map_highlight_outline)
    .resolve_scale(color='independent')
    .project('equalEarth')
    .properties(
        width=720,
        height=420,
        title=''
    )
)

# ------------------------
# Final layout (Q2)
# ------------------------

q2_final = alt.hconcat(
    left_col, map_chart, spacing=16
).add_params(
    delta_metric_sel, country_pick, region_pick
).properties(
    title={
        "text": "Q2: Which countries have experienced the greatest changes as visa-free countries between 2006 and 2021?",
        "subtitle": "Disclaimer: visa-free count for years 2007 and 2009 has been interpolated."
    }
)

covid_trend = (
  alt.Chart(covid_vfc)
  .mark_line(point=True)
  .encode(
    x=alt.X('year:O', axis=alt.Axis(labelAngle=-45), title=''),
    y=alt.Y('mean(visa_free_count):Q', title=''),
    color=alt.Color('region:N', title='Region', legend=None),
  )
  .properties(
      width=700,
      title='Trend by region of mean visa-free destinations'
  )
)

cv_bar_top = (
    alt.Chart(cv_max_delta_df)
    .mark_bar()
    .transform_window(
        rank='rank(abs_delta)',  # max YoY per country
        sort=[{'field': 'abs_delta', 'order': 'descending'}]
    )
    .transform_filter(
        alt.datum.rank <= 6
    )
    .encode(
        x=alt.X(
            'code3:N',
            title='',
            sort=alt.SortField(field='rank', order='ascending'),
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y('delta:Q', title=''),
        color=alt.Color(
            'region:N',
            legend=None,
            scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE)
        ),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('delta:Q', title='Max YoY delta', format='.0f')
        ]
    )
    .properties(
        width=260,
        height=200,
        title='Top 10 countries by max delta YoY (2020 - 2022)'
    )
)
# cv_bar_top

cv_bar_top = cv_bar_top.properties(width=520, height=220)
covid_trend = covid_trend.properties(width=520, height=220)

# ------------------------
# Final layout (Q3)
# ------------------------
q3_final = alt.hconcat(
    cv_bar_top, covid_trend, spacing=16
).properties(
    title='Q3: What was the impact of COVID-19 on visa-free mobility?'
)

alliace_chart = alt.Chart(f_long_us_access).mark_bar().encode(
    x=alt.X(
        'alliance:N',
        title='',
        sort='-y',
        axis=alt.Axis(labelAngle=0)
    ),
    y=alt.Y('count()', title='Number of Countries'),
    color=alt.Color(
        'us_visa_free_label:N',
        scale=alt.Scale(domain=['Visa Free', 'Not Visa Free'], range=['#2f78b3', '#df745e']),
        legend=alt.Legend(title='Visa Status')
    ),
    tooltip=[
        alt.Tooltip('alliance:N', title='Alliance'),
        alt.Tooltip('us_visa_free_label:N', title='Visa status'),
        alt.Tooltip('count():Q', title='# Countries')
    ]
).properties(
    width=320,
    height=180,
    title='Visa-Free Access by Alliance'
)
# alliace_chart

filter_options = [
    'All countries',
    'High income',
    'TIAR members',
    'CSTO members',
    'NATO members'
]

filter_param = alt.param(
    name='filter',
    value='All countries',
    bind=alt.binding_select(options=filter_options, name='Filter economy/alliance: ')
)

hi_df = complete_us_access[complete_us_access['inc_group'] == 'High income'].copy()

donut = alt.Chart(hi_df).mark_arc(innerRadius=50).encode(
    theta=alt.Theta('count()', title=''),
    color=alt.Color(
        'us_visa_free_label:N',
        scale=alt.Scale(domain=['Visa Free', 'Not Visa Free'], range=['#2f78b3', '#df745e']),
        legend=None
    ),
    tooltip=[
        alt.Tooltip("us_visa_free_label:N", title="Visa Status"),
        alt.Tooltip("count():Q", title="# Countries")
    ]
).properties(
    width=320,
    height=180,
    title="High Income Countries: Visa Free vs Not Visa Free"
)
# donut

# ------------------------
# Map (right)
# ------------------------

map_base = alt.Chart(world).transform_calculate(
    lookup_key='datum.properties.ADM0_A3'
).transform_lookup(
    lookup='lookup_key',
    from_=alt.LookupData(
        complete_us_access,
        key='code3',
        fields=['country','inc_group','us_visa_free_flag',
                'us_visa_free_label','is_nato_member','is_tiar_member',
                'is_csto_member']
    )
)

map_color_layer = (
    map_base
    .mark_geoshape()
    .transform_filter(
        alt.datum.us_visa_free_flag != None
    )
    .encode(
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.6),
        color=alt.Color(
            'us_visa_free_label:N',
            scale=alt.Scale(domain=["Not Visa Free", "Visa Free"], range=['#df745e','#2f78b3']),
            legend=None
        ),
        opacity = alt.condition(
              (
                  (filter_param == 'All countries')
                  | ((filter_param == 'High income') & (alt.datum.inc_group == 'High income'))
                  | ((filter_param == 'TIAR members') & (alt.datum.is_tiar_member == 1))
                  | ((filter_param == 'CSTO members') & (alt.datum.is_csto_member == 1))
                  | ((filter_param == 'NATO members') & (alt.datum.is_nato_member == 1))
              ),
              alt.value(1.0),
              alt.value(0.25)
          ),

        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('us_visa_free_label:N', title='Access')
        ]
    )
    .add_params(filter_param)
)

map_chart = (
    map_color_layer
    .project('equalEarth')
    .properties(
        width=720,
        height=420,
        title=''
    )
)
# map_chart

left_col = alt.vconcat(
    alliace_chart,
    donut,
    spacing=16
).resolve_scale(color='independent')

# ------------------------
# Final layout (Q4)
# ------------------------
q4_final = alt.hconcat(
    left_col, map_chart, spacing=0
).properties(
    title='Q4: Do countries that belong to certain global alliances, or have stronger economies, tend to enjoy greater visa-free access?'
).resolve_scale(color='independent')

# q4_final

left_col = alt.vconcat(q1_final, q2_final)
right_col = alt.vconcat(q3_final, q4_final)
final_dashboard = alt.hconcat(left_col, right_col)

final_dashboard = (
    final_dashboard
    .configure_axis(labelFontSize=12, titleFontSize=12)
    .configure_legend(labelFontSize=12, titleFontSize=12)
    .configure_view(stroke=None)
)

# final_dashboard