# This Streamlit app implements a simple market analysis dashboard inspired by
# Smart‑Money‑Concepts (SMC) trading.  Users can select an asset type
# (crypto, forex or stock) and a symbol, pull recent price data via the
# yfinance library and visualise the market structure, potential Fair Value
# Gaps (FVGs), swing highs/lows, breaks of structure (BOS/CHoCH) and
# simplistic order blocks.  The app leverages the trading definitions
# described by reputable sources: a swing high occurs when a price peak is
# flanked by lower highs on either side【172045438186832†L130-L136】, and a
# swing low forms when a trough is flanked by higher lows【172045438186832†L130-L136】.  An
# order block is identified as the last opposing candle before a sharp
# reversal【46680729103635†L226-L233】.  Fair value gaps are detected when the wicks of
# the first and third candle in a three‑candle sequence do not overlap,
# leaving a price void that may later be revisited【904932770475045†L63-L83】.

import datetime
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


@dataclass
class SwingPoints:
    """Stores the indices of swing highs and swing lows."""
    highs: List[int]
    lows: List[int]


def detect_swings(df: pd.DataFrame, order: int = 2) -> SwingPoints:
    """Detect swing highs and lows in a price series.

    A swing high is defined as a bar whose high is greater than the highs
    of a specified number of bars on both sides【172045438186832†L130-L136】.  A swing
    low is defined as a bar whose low is less than the lows of the same
    number of neighbouring bars.  The `order` parameter controls how many
    bars on each side are compared.

    Args:
        df: DataFrame with 'High' and 'Low' columns.
        order: Number of neighbouring bars to compare on each side.

    Returns:
        SwingPoints instance containing lists of indices for swing highs
        and swing lows.
    """
    highs: List[int] = []
    lows: List[int] = []
    for i in range(order, len(df) - order):
        window_high = df['High'].iloc[i - order:i + order + 1]
        window_low = df['Low'].iloc[i - order:i + order + 1]
        # Extract scalar values for the current bar.  Using `.item()` converts the
        # single‑element Series returned by `.iloc` into a plain Python float,
        # which avoids the ambiguous truth value errors raised when comparing
        # pandas Series inside an `if` statement【52063707001768†L35-L56】.
        current_high_val: float = float(df['High'].iloc[i])
        current_low_val: float = float(df['Low'].iloc[i])
        max_high_val: float = float(window_high.max())
        min_low_val: float = float(window_low.min())
        # A swing high occurs when the current high equals the maximum in the
        # window and is strictly greater than all other highs in that window.
        if current_high_val == max_high_val:
            # Drop the current bar by label to ensure we are comparing against
            # the remaining values only.  We use window_high.index to find the
            # current bar’s label rather than df.index[i] to handle cases where
            # the DataFrame’s index is not a simple RangeIndex.
            other_highs = window_high.drop(index=window_high.index[order])
            if float(other_highs.max()) < current_high_val:
                highs.append(i)
        # A swing low occurs when the current low equals the minimum in the
        # window and is strictly lower than all other lows in that window.
        if current_low_val == min_low_val:
            other_lows = window_low.drop(index=window_low.index[order])
            if float(other_lows.min()) > current_low_val:
                lows.append(i)
    return SwingPoints(highs=highs, lows=lows)


def detect_fvg(df: pd.DataFrame) -> List[Tuple[str, int, int]]:
    """Identify Fair Value Gaps (FVGs) in a price series.

    A bullish FVG occurs when the low of the candle two bars ahead is
    greater than the high of the candle two bars behind, leaving an
    untraded space in between【904932770475045†L63-L83】.  Conversely, a bearish FVG
    occurs when the high of the forward candle is less than the low of
    the backward candle.  This function scans the data and returns a list
    of tuples describing the type of gap and the start/end indices.

    Args:
        df: DataFrame with 'High' and 'Low' columns.

    Returns:
        A list of tuples: ("bullish" or "bearish", left_index, right_index).
    """
    gaps: List[Tuple[str, int, int]] = []
    for i in range(1, len(df) - 1):
        left_high = df['High'].iloc[i - 1]
        left_low = df['Low'].iloc[i - 1]
        right_high = df['High'].iloc[i + 1]
        right_low = df['Low'].iloc[i + 1]
        # Bullish FVG: gap below price; price may later return downwards into the gap
        if right_low > left_high:
            gaps.append(('bullish', i - 1, i + 1))
        # Bearish FVG: gap above price; price may later return upwards into the gap
        elif right_high < left_low:
            gaps.append(('bearish', i - 1, i + 1))
    return gaps


def detect_bos(df: pd.DataFrame, swings: SwingPoints) -> List[Tuple[int, str]]:
    """Detect Breaks Of Structure (BOS) and Changes of Character (CHoCH).

    A bullish BOS is flagged when price closes above the prior swing high
    with a decisive body close【172045438186832†L143-L146】.  A bearish BOS occurs
    when price closes below the prior swing low.  A Change of Character
    (CHoCH) is marked when price breaks the opposite swing level against
    the current trend.  The current trend is estimated by comparing the
    most recent swing highs and lows.

    Args:
        df: DataFrame with 'Close' column.
        swings: SwingPoints returned from detect_swings().

    Returns:
        A list of tuples (index, 'BOS' or 'CHoCH').
    """
    signals: List[Tuple[int, str]] = []
    # Determine recent trend based on the last two swing highs and lows
    trend: str = 'unknown'
    if len(swings.highs) >= 2 and len(swings.lows) >= 2:
        # Use the latest two swings to infer trend
        last_high = swings.highs[-1]
        prev_high = swings.highs[-2]
        last_low = swings.lows[-1]
        prev_low = swings.lows[-2]
        if df['High'].iloc[last_high] > df['High'].iloc[prev_high] and df['Low'].iloc[last_low] > df['Low'].iloc[prev_low]:
            trend = 'up'
        elif df['High'].iloc[last_high] < df['High'].iloc[prev_high] and df['Low'].iloc[last_low] < df['Low'].iloc[prev_low]:
            trend = 'down'
    # Check each swing level to see if it has been broken
    last_high_broken: bool = False
    last_low_broken: bool = False
    for i in range(max(max(swings.highs or [0]), max(swings.lows or [0])) + 1, len(df)):
        close_price = df['Close'].iloc[i]
        # Break of structure on the bullish side
        if not last_high_broken and swings.highs:
            last_high_idx = swings.highs[-1]
            if close_price > df['High'].iloc[last_high_idx]:
                if trend == 'up':
                    signals.append((i, 'BOS'))
                else:
                    signals.append((i, 'CHoCH'))
                last_high_broken = True
        # Break of structure on the bearish side
        if not last_low_broken and swings.lows:
            last_low_idx = swings.lows[-1]
            if close_price < df['Low'].iloc[last_low_idx]:
                if trend == 'down':
                    signals.append((i, 'BOS'))
                else:
                    signals.append((i, 'CHoCH'))
                last_low_broken = True
        if last_high_broken and last_low_broken:
            break
    return signals


def detect_order_blocks(df: pd.DataFrame, look_ahead: int = 3, threshold: float = 1.5) -> List[Tuple[str, int]]:
    """Identify simplistic order blocks in a price series.

    This is a pragmatic interpretation of the academic definition: an order
    block is the last opposing candle before a significant directional
    movement【46680729103635†L226-L233】.  We scan each candle and look ahead a
    fixed number of bars; if price moves more than a multiple of the average
    range (|close-open|) in the opposite direction, the current candle is
    flagged as an order block.  Bullish order blocks correspond to last
    bearish candles before an upswing and bearish order blocks correspond to
    last bullish candles before a downswing.

    Args:
        df: DataFrame with 'Open' and 'Close' columns.
        look_ahead: Number of bars to look ahead when measuring the move.
        threshold: Multiple of average range required to qualify as a significant
                   move.

    Returns:
        List of tuples ("bullish" or "bearish", index).
    """
    avg_range = df['Close'].sub(df['Open']).abs().rolling(window=look_ahead).mean()
    ob_list: List[Tuple[str, int]] = []
    for i in range(len(df) - look_ahead):
        current_range = avg_range.iloc[i] or 1e-9
        # Skip until average range is defined
        if np.isnan(current_range):
            continue
        move = df['Close'].iloc[i + look_ahead] - df['Open'].iloc[i + 1]
        # Bullish movement: look for last bearish candle
        if df['Close'].iloc[i] < df['Open'].iloc[i] and move > threshold * current_range:
            ob_list.append(('bullish', i))
        # Bearish movement: look for last bullish candle
        if df['Close'].iloc[i] > df['Open'].iloc[i] and move < -threshold * current_range:
            ob_list.append(('bearish', i))
    return ob_list


def plot_chart(df: pd.DataFrame, swings: SwingPoints, fvgs: List[Tuple[str, int, int]],
               bos_signals: List[Tuple[int, str]], order_blocks: List[Tuple[str, int]]) -> go.Figure:
    """Create an interactive candlestick chart with technical overlays.

    The chart includes markers for swing highs/lows, shaded regions for fair
    value gaps, vertical lines for BOS/CHoCH signals, and rectangles for
    simplistic order blocks.  It uses Plotly for interactivity.

    Args:
        df: DataFrame with OHLC data.
        swings: SwingPoints containing indices of swing highs/lows.
        fvgs: List of FVG tuples.
        bos_signals: List of BOS/CHoCH tuples (index, label).
        order_blocks: List of (type, index) order blocks.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    # Swing highs/lows markers
    fig.add_trace(go.Scatter(
        x=df.index[swings.highs],
        y=df['High'].iloc[swings.highs],
        mode='markers',
        marker=dict(size=8, symbol='triangle-up'),
        name='Swing Highs'
    ))
    fig.add_trace(go.Scatter(
        x=df.index[swings.lows],
        y=df['Low'].iloc[swings.lows],
        mode='markers',
        marker=dict(size=8, symbol='triangle-down'),
        name='Swing Lows'
    ))
    # Fair value gaps shading
    for gap_type, left_idx, right_idx in fvgs:
        color = 'rgba(0, 255, 0, 0.2)' if gap_type == 'bullish' else 'rgba(255, 0, 0, 0.2)'
        fig.add_shape(
            type='rect',
            x0=df.index[left_idx], y0=df['Low'].min(),
            x1=df.index[right_idx], y1=df['High'].max(),
            fillcolor=color,
            line=dict(width=0),
            layer='below',
            name=f'{gap_type.capitalize()} FVG'
        )
    # BOS/CHoCH vertical lines
    for idx, label in bos_signals:
        fig.add_shape(
            type='line',
            x0=df.index[idx], y0=df['Low'].min(),
            x1=df.index[idx], y1=df['High'].max(),
            line=dict(dash='dot'),
            name=label
        )
    # Order block rectangles
    for block_type, idx in order_blocks:
        candle_low = df['Low'].iloc[idx]
        candle_high = df['High'].iloc[idx]
        color = 'rgba(0, 0, 255, 0.2)' if block_type == 'bullish' else 'rgba(255, 165, 0, 0.2)'
        fig.add_shape(
            type='rect',
            x0=df.index[idx], y0=candle_low,
            x1=df.index[idx + 1] if idx + 1 < len(df) else df.index[idx], y1=candle_high,
            fillcolor=color,
            line=dict(width=0),
            layer='below',
            name=f'{block_type.capitalize()} OB'
        )
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(orientation='h', y=1.05)
    )
    return fig


def main():
    st.set_page_config(page_title='SMC Market Analyzer', layout='wide')
    st.title('Smart Money Concepts Market Analyzer')
    st.markdown(
        """
        Use this tool to explore the current market behaviour of cryptocurrencies, forex
        pairs and stocks through the lens of Smart Money Concepts (SMC).  The app
        fetches historical price data from Yahoo Finance, identifies key market
        structure points (swing highs/lows), highlights potential Fair Value Gaps
        (FVGs)【904932770475045†L63-L83】, breaks of structure (BOS) and changes of character
        (CHoCH)【172045438186832†L143-L146】, and marks simple order blocks【46680729103635†L226-L233】.  This
        educational tool is intended for analysis only and should not be taken
        as financial advice.
        """
    )

    # Sidebar controls
    asset_type = st.sidebar.selectbox('Asset type', ['Crypto', 'Forex', 'Stock'])
    # Predefined symbols for convenience
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD']
    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCHF=X']
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'SPY']
    if asset_type == 'Crypto':
        symbols = crypto_symbols
    elif asset_type == 'Forex':
        symbols = forex_symbols
    else:
        symbols = stock_symbols
    symbol = st.sidebar.selectbox('Symbol', symbols)
    period = st.sidebar.selectbox('Period', ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
    interval = st.sidebar.selectbox('Interval', ['1d', '1h', '30m'], index=0)

    # Fetch data using yfinance
    @st.cache_data
    def load_data(sym: str, per: str, interv: str) -> pd.DataFrame:
        df = yf.download(sym, period=per, interval=interv, progress=False)
        df.dropna(inplace=True)
        return df
    try:
        df = load_data(symbol, period, interval)
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return
    if df.empty:
        st.warning('No data returned for the selected parameters.')
        return

    st.subheader(f'{symbol} price action')
    # Detect technical features
    swings = detect_swings(df)
    fvgs = detect_fvg(df)
    bos_signals = detect_bos(df, swings)
    order_blocks = detect_order_blocks(df)
    fig = plot_chart(df, swings, fvgs, bos_signals, order_blocks)
    st.plotly_chart(fig, use_container_width=True)

    # Summary section
    st.subheader('Summary')
    summary_lines: List[str] = []
    # Determine market bias
    bias: str = 'Unknown'
    if swings.highs and swings.lows:
        if len(swings.highs) >= 2 and len(swings.lows) >= 2:
            last_high = swings.highs[-1]
            prev_high = swings.highs[-2]
            last_low = swings.lows[-1]
            prev_low = swings.lows[-2]
            if df['High'].iloc[last_high] > df['High'].iloc[prev_high] and df['Low'].iloc[last_low] > df['Low'].iloc[prev_low]:
                bias = 'Bullish'
            elif df['High'].iloc[last_high] < df['High'].iloc[prev_high] and df['Low'].iloc[last_low] < df['Low'].iloc[prev_low]:
                bias = 'Bearish'
            else:
                bias = 'Range/Indecision'
    summary_lines.append(f'**Trend Bias:** {bias}')
    summary_lines.append(f'**Number of swing highs:** {len(swings.highs)}')
    summary_lines.append(f'**Number of swing lows:** {len(swings.lows)}')
    summary_lines.append(f'**Number of Fair Value Gaps detected:** {len(fvgs)}')
    summary_lines.append(f'**Number of Order Blocks detected:** {len(order_blocks)}')
    if bos_signals:
        last_signal_idx, last_signal_type = bos_signals[-1]
        summary_lines.append(f'**Latest structural event:** {last_signal_type} on {df.index[last_signal_idx].date()}')
    else:
        summary_lines.append('**Latest structural event:** None detected in the selected period.')
    st.markdown('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
