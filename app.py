"""Monthly IMF bilateral real and nominal exchange rates; Python 3.12."""
from datetime import date
import os
import re
import time
import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE = 'https://api.imf.org/external/sdmx/3.0'
FIRST = '1976-01'


def get_json(path):
    headers = {'Accept': 'application/json'}
    if os.getenv('IMF_API_KEY'):
        headers['Ocp-Apim-Subscription-Key'] = os.environ['IMF_API_KEY']
    for attempt in range(3):
        try:
            response = requests.get(BASE + path, headers=headers, timeout=60)
            if response.status_code in (204, 404):
                return {}
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.HTTPError as exc:
            code = exc.response.status_code
            if code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f'IMF returned HTTP {code}. Try again later; if access is restricted, set IMF_API_KEY.') from exc
        except requests.RequestException as exc:
            if attempt == 2:
                raise RuntimeError('IMF could not be reached. Check the connection and retry.') from exc
        time.sleep(2 ** attempt)
    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def countries():
    payload = get_json('/structure/dataflow/IMF.STA/CPI/+?references=all')
    for cl in payload.get('data', {}).get('codelists', []):
        if cl['id'] == 'CL_COUNTRY':
            result = {c['id']: c.get('names', {}).get('en', c.get('name', c['id']))
                      for c in cl['codes'] if re.fullmatch(r'[A-Z]{3}', c['id']) or c['id'] == 'G163'}
            result['G163'] = 'Euro area'
            return result
    raise RuntimeError('IMF country metadata is unavailable. Please retry later.')


def parse_series(payload):
    """Decode SDMX indexed observation dimensions, including IMF YYYY-Mmm dates."""
    data = payload.get('data', payload)
    records = {}
    for ds in data.get('dataSets', []):
        structure = data['structures'][ds.get('structure', 0)]
        dimensions = structure['dimensions']['observation']
        position = next(i for i, d in enumerate(dimensions) if d['id'] == 'TIME_PERIOD')
        periods = dimensions[position]['values']
        series = ds.get('series', {})
        if len(series) > 1:
            raise ValueError('IMF returned multiple series for a single key; refusing to combine them.')
        for item in series.values():
            for key, obs in item.get('observations', {}).items():
                entry = periods[int(key.split(':')[position])]
                raw = entry.get('id', entry.get('value', ''))
                month = re.sub(r'^(\d{4})-M(\d{2})$', r'\1-\2', raw)
                if not re.fullmatch(r'\d{4}-\d{2}', month):
                    continue
                value = pd.to_numeric(obs[0], errors='coerce')
                records[pd.Period(month, freq='M')] = value
    result = pd.Series(records, dtype=float).sort_index()
    return result.where(np.isfinite(result) & (result > 0))


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_series(country, kind):
    flow, key = ('CPI', f'{country}.CPI._T.IX.M') if kind == 'cpi' else ('ER', f'{country}.XDC_USD.PA_RT.M')
    return parse_series(get_json(f'/data/dataflow/IMF.STA/{flow}/+/{key}'))


def calculate(p1, p2, e1, e2, start, end):
    frame = pd.DataFrame(index=pd.period_range(start, end, freq='M'))
    for name, series in [('P1', p1), ('P2', p2), ('XDC_USD_1', e1), ('XDC_USD_2', e2)]:
        frame[name] = series.reindex(frame.index)
        frame[name] = frame[name].where(np.isfinite(frame[name]) & (frame[name] > 0))
    frame['S'] = frame.XDC_USD_2 / frame.XDC_USD_1
    frame['RER'] = frame.S * frame.P1 / frame.P2
    for name in ['S', 'RER']:
        frame[name] = frame[name].where(np.isfinite(frame[name]) & (frame[name] > 0))
    frame['valid_real'] = frame.RER.notna()
    mean = frame.RER.mean()
    frame['RER_star'] = mean
    frame['Real_Index'] = 100 * frame.RER / mean if pd.notna(mean) else np.nan
    frame.index.name = 'month'
    return frame


def main():
    st.set_page_config(page_title='Real Exchange Rate Explorer', page_icon='🌍', layout='wide')
    st.title('Real Exchange Rate Explorer')
    st.caption('Monthly IMF data · January 1976 onward · A sample average of 100')
    try:
        with st.spinner('Loading IMF countries…'):
            options = countries()
    except Exception as exc:
        st.error(str(exc))
        if st.button('Retry IMF connection'):
            st.cache_data.clear()
            st.rerun()
        return
    codes = sorted(options, key=options.get)
    with st.sidebar:
        st.header('Choose your sample')
        c1 = st.selectbox('Country 1', codes, index=codes.index('USA'), format_func=lambda c: f'{options[c]} ({c})')
        c2 = st.selectbox('Country 2', codes, index=codes.index('GBR'), format_func=lambda c: f'{options[c]} ({c})')
        months = pd.period_range(FIRST, date.today(), freq='M').astype(str).tolist()
        start = st.selectbox('Start month', months, index=0)
        end = st.selectbox('End month', months, index=len(months)-1)
        view = st.radio('Show', ['Real', 'Nominal', 'Both'], index=2, horizontal=True)
        if st.button('Refresh IMF data'):
            st.cache_data.clear()
            st.rerun()
    if start > end:
        st.error('Start month must be on or before end month.')
        return
    try:
        with st.spinner('Retrieving monthly CPI and exchange rates…'):
            p1, p2 = fetch_series(c1, 'cpi'), fetch_series(c2, 'cpi')
            e1, e2 = fetch_series(c1, 'er'), fetch_series(c2, 'er')
            frame = calculate(p1, p2, e1, e2, start, end)
    except Exception as exc:
        st.error(f'Unable to load the sample: {exc}')
        return
    units = f'{options[c2]} currency units per 1 {options[c1]} currency unit'
    st.write(f'**{options[c1]} → {options[c2]}**')
    st.caption(f'Nominal S: {units}. IMF series: monthly period-average local currency per USD.')
    a, b, c = st.columns(3)
    a.metric('Selected months', len(frame))
    b.metric('Valid real observations', int(frame.valid_real.sum()))
    mean = frame.RER.mean()
    c.metric('Sample average RER*', f'{mean:.6g}' if pd.notna(mean) else 'Unavailable')
    if not frame.valid_real.all():
        st.warning('Some months lack valid CPI or exchange-rate data. Gaps are left blank; no interpolation or forward filling is used. Nominal observations remain available when CPI is missing.')
    if not frame.valid_real.any():
        st.info('No overlapping valid data for a real index. Try different countries or dates.')
    fig = make_subplots(specs=[[{'secondary_y': view == 'Both'}]])
    x = frame.index.to_timestamp()
    if view in ('Real', 'Both'):
        fig.add_trace(go.Scatter(x=x, y=frame.Real_Index, name='Real index', connectgaps=False, line={'color': '#167c80'}))
        fig.add_hline(y=100, line_dash='dot', line_color='#8a959f')
        fig.update_yaxes(title_text='Real index (sample mean = 100)', secondary_y=False)
    if view in ('Nominal', 'Both'):
        fig.add_trace(go.Scatter(x=x, y=frame.S, name='Nominal S', connectgaps=False, line={'color': '#d57c32'}), secondary_y=view == 'Both')
        fig.update_yaxes(title_text=f'Nominal S: {c2} currency / {c1} currency', secondary_y=view == 'Both')
    fig.update_layout(height=520, hovermode='x unified', legend={'orientation': 'h'}, margin={'t': 30}, xaxis_title='Month')
    st.plotly_chart(fig, use_container_width=True)
    export = frame.reset_index()
    export['month'] = export['month'].astype(str)
    export.insert(1, 'country_1', c1)
    export.insert(2, 'country_2', c2)
    st.download_button('Download sample CSV', export.to_csv(index=False).encode('utf-8'), f'rer_{c1}_{c2}_{start}_{end}.csv', 'text/csv')
    with st.expander('Observations and calculations'):
        st.dataframe(export, use_container_width=True)
    with st.expander('Method and interpretation', expanded=True):
        st.markdown('S = (Country 2 currency/USD) ÷ (Country 1 currency/USD).  '
                    'RER = S × P1 ÷ P2. RER* is the arithmetic average over valid months in the selected sample.  '
                    'Real Index = 100 × RER ÷ RER*. Changing either date recalculates the average. '
                    'An increase indicates real appreciation of Country 1 relative to Country 2 under this convention. '
                    'The nominal series uses actual currency units. CPI base years affect raw RER levels; '
                    'the normalized index is invariant to constant CPI rebasing. A value above 100 alone does not establish overvaluation. '
                    'Currency reforms, changing currency regimes, and methodological breaks can affect comparisons; '
                    'this app does not splice predecessor currencies or adjust breaks. Euro area uses IMF code G163; earlier data may be absent.')
        st.markdown('[IMF CPI](https://data.imf.org/en/datasets/IMF.STA%3ACPI) · '
                    '[IMF exchange rates](https://data.imf.org/en/datasets/IMF.STA%3AER) · '
                    '[IMF API](https://data.imf.org/en/Resource-Pages/IMF-API)')


if __name__ == '__main__':
    main()
