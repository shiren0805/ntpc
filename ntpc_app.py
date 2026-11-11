# -*- coding: utf-8 -*-
"""新北市政府文化局｜活動與公告資料視覺化分析 (Streamlit)

資料來源：新北市政府資料開放平台
https://data.ntpc.gov.tw/api/datasets/781b822e-214a-4b9a-b4db-32c9f4626d98/csv/file

執行方式：
    pip install -r requirements.txt
    streamlit run ntpc_app.py
"""
import io
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="新北市文化局活動分析", page_icon="🎭", layout="wide")

DATA_URL = "https://data.ntpc.gov.tw/api/datasets/781b822e-214a-4b9a-b4db-32c9f4626d98/csv/file"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

COLUMN_MAP = {
    "author": "發布單位",
    "type": "類型",
    "startdate": "開始日期",
    "enddate": "結束日期",
    "title": "標題",
    "link": "連結",
    "description": "內容",
    "pubdate": "公告日期",
}
STATUS_ORDER = ["進行中", "尚未開始", "已結束", "日期不明"]
STATUS_COLORS = {
    "進行中": "#16A34A",
    "尚未開始": "#2563EB",
    "已結束": "#9CA3AF",
    "日期不明": "#F59E0B",
}


# ---------------------------------------------------------------- 資料讀取
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    """下載 CSV 並整理欄位（內容欄含逗號與換行，交給 pandas 的引號解析處理）。"""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig", dtype=str)
    df = df.rename(columns=COLUMN_MAP)

    for col in ["開始日期", "結束日期", "公告日期"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["類型", "發布單位"]:
        if col in df.columns:
            df[col] = df[col].fillna("未分類").str.strip()
    df["標題"] = df["標題"].fillna("").str.strip()
    df["內容"] = df["內容"].fillna("")
    return df.drop_duplicates().reset_index(drop=True)


def add_status(df: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    s, e = df["開始日期"], df["結束日期"]
    e = e.fillna(s)  # 沒有結束日期就視為單日
    status = pd.Series("進行中", index=df.index)
    status[e < ref] = "已結束"
    status[s > ref] = "尚未開始"
    status[s.isna()] = "日期不明"
    df["狀態"] = pd.Categorical(status, categories=STATUS_ORDER, ordered=True)

    days = (e - s).dt.days + 1
    df["天數"] = days.where(days > 0)
    df["開始月份"] = s.dt.strftime("%Y-%m").fillna("日期不明")
    return df


# ---------------------------------------------------------------- 圖表
def pie_type(df: pd.DataFrame):
    counts = df["類型"].value_counts().reset_index()
    counts.columns = ["類型", "數量"]
    fig = px.pie(counts, names="類型", values="數量", hole=0.35, title="各類型占比")
    fig.update_traces(textinfo="percent+label", sort=False)
    fig.update_layout(height=480, showlegend=False, title_font_size=20)
    return fig


def bar_status_by_type(df: pd.DataFrame):
    g = df.groupby(["類型", "狀態"], observed=True).size().reset_index(name="數量")
    fig = px.bar(
        g, x="類型", y="數量", color="狀態", barmode="stack",
        color_discrete_map=STATUS_COLORS,
        category_orders={"狀態": STATUS_ORDER},
        title="各類型的活動狀態",
    )
    fig.update_layout(height=480, title_font_size=20)
    return fig


def line_trend(df: pd.DataFrame, basis: str, freq_label: str, by_type: bool):
    d = df.dropna(subset=[basis]).copy()
    if d.empty:
        return None
    period = "M" if freq_label == "月" else "W"
    d["期間"] = d[basis].dt.to_period(period).dt.start_time
    step = "MS" if period == "M" else "7D"

    if by_type:
        g = d.groupby(["期間", "類型"]).size().reset_index(name="數量")
        fig = px.line(g, x="期間", y="數量", color="類型", markers=True)
    else:
        g = d.groupby("期間").size()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq=step), fill_value=0)
        g = g.rename_axis("期間").reset_index(name="數量")
        fig = px.line(g, x="期間", y="數量", markers=True)
        avg = g["數量"].mean()
        fig.add_hline(
            y=avg, line_dash="dash", line_color="red",
            annotation_text=f"平均 {avg:.1f}", annotation_position="top left",
        )
    fig.update_layout(
        height=500, title=f"每{freq_label}件數趨勢（依「{basis}」）", title_font_size=20,
        xaxis_title=basis, yaxis_title="件數",
    )
    return fig


def sunburst(df: pd.DataFrame, depth: str):
    path = ["類型", "狀態"] if depth == "類型 → 狀態" else ["類型", "狀態", "開始月份"]
    d = df.copy()
    d["狀態"] = d["狀態"].astype(str)
    agg = d.groupby(path).size().reset_index(name="數量")
    fig = px.sunburst(agg, path=path, values="數量", title=f"{depth} 分布")
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(height=700, title_font_size=20)
    return fig


def gantt(df: pd.DataFrame, ref: pd.Timestamp, top: int):
    d = df.dropna(subset=["開始日期"]).copy()
    d["結束"] = d["結束日期"].fillna(d["開始日期"])
    d = d[d["結束"] >= d["開始日期"]]
    if d.empty:
        return None
    d = d.sort_values("開始日期", ascending=False).head(top)
    d["結束"] = d["結束"] + pd.Timedelta(days=1)  # 單日活動也要有長度
    d["標題簡稱"] = d["標題"].str.slice(0, 28)
    fig = px.timeline(
        d, x_start="開始日期", x_end="結束", y="標題簡稱", color="狀態",
        color_discrete_map=STATUS_COLORS, hover_name="標題",
        hover_data={"類型": True, "標題簡稱": False, "結束": False},
        title=f"活動時間軸（開始日期最近的 {len(d)} 筆）",
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.add_shape(
        type="line", x0=ref, x1=ref, y0=0, y1=1, yref="paper",
        line=dict(color="red", dash="dash"),
    )
    fig.add_annotation(
        x=ref, y=1, yref="paper", text="基準日", showarrow=False,
        font=dict(color="red"), yanchor="bottom",
    )
    fig.update_layout(height=max(500, 22 * len(d) + 150), title_font_size=20)
    return fig


def duration_chart(df: pd.DataFrame):
    d = df.dropna(subset=["天數"])
    d = d[d["天數"] <= 366]  # 排除跨年度的長期公告，避免壓縮圖形
    if d.empty:
        return None
    fig = px.box(
        d, x="類型", y="天數", color="類型", points="outliers",
        hover_name="標題", title="各類型活動 / 公告天數（排除超過 1 年者）",
    )
    fig.update_layout(height=480, showlegend=False, title_font_size=20)
    return fig


# ---------------------------------------------------------------- 介面
st.title("🎭 新北市文化局｜活動與公告分析")
st.caption("資料來源：新北市政府資料開放平台（文化局活動與公告）")

with st.sidebar:
    st.header("資料")
    if st.button("🔄 重新下載資料", use_container_width=True):
        load_data.clear()
    ref_date = st.date_input("以此日期判斷活動狀態", value=date.today())

try:
    with st.spinner("下載資料中…"):
        raw = load_data(DATA_URL)
except requests.RequestException as e:
    st.error(f"下載失敗：{e}")
    st.stop()
except Exception as e:
    st.error(f"資料解析失敗：{e}")
    st.stop()

if raw.empty:
    st.warning("資料集為空。")
    st.stop()

ref = pd.Timestamp(ref_date)
data = add_status(raw, ref)

with st.sidebar:
    st.header("篩選條件")
    types = sorted(data["類型"].unique())
    sel_types = st.multiselect("類型", types, default=types)
    sel_status = st.multiselect("狀態", STATUS_ORDER, default=STATUS_ORDER)
    keyword = st.text_input("標題 / 內容關鍵字")

    st.header("圖表設定")
    basis = st.radio("折線圖時間基準", ["開始日期", "公告日期"], horizontal=True)
    freq_label = st.radio("折線圖統計單位", ["月", "週"], horizontal=True)
    by_type = st.checkbox("折線圖依類型分線", value=False)
    sun_depth = st.radio("旭日圖層級", ["類型 → 狀態", "類型 → 狀態 → 開始月份"])
    top_gantt = st.slider("時間軸顯示筆數", 10, 100, 40, 5)

df = data[data["類型"].isin(sel_types) & data["狀態"].isin(sel_status)]
if keyword.strip():
    kw = keyword.strip()
    mask = df["標題"].str.contains(kw, case=False, na=False, regex=False) | df[
        "內容"
    ].str.contains(kw, case=False, na=False, regex=False)
    df = df[mask]

if df.empty:
    st.warning("目前篩選條件下沒有資料。")
    st.stop()

# KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("總筆數", f"{len(df):,}")
c2.metric("類型數", df["類型"].nunique())
c3.metric("進行中", int((df["狀態"] == "進行中").sum()))
c4.metric("尚未開始", int((df["狀態"] == "尚未開始").sum()))
c5.metric("已結束", int((df["狀態"] == "已結束").sum()))

tab_over, tab_line, tab_sun, tab_gantt, tab_data = st.tabs(
    ["🥧 總覽", "📈 趨勢折線圖", "☀️ 旭日圖", "🗓️ 時間軸", "📋 資料表"]
)

with tab_over:
    left, right = st.columns(2)
    left.plotly_chart(pie_type(df), use_container_width=True)
    right.plotly_chart(bar_status_by_type(df), use_container_width=True)
    fig = duration_chart(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with tab_line:
    fig = line_trend(df, basis, freq_label, by_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"「{basis}」沒有可用的日期資料。")

with tab_sun:
    st.plotly_chart(sunburst(df, sun_depth), use_container_width=True)

with tab_gantt:
    fig = gantt(df, ref, top_gantt)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("紅色虛線為基準日；單日活動以 1 天的長度顯示。")
    else:
        st.info("沒有可繪製的日期資料。")

with tab_data:
    show = df[["類型", "標題", "開始日期", "結束日期", "狀態", "公告日期", "連結"]].sort_values(
        "公告日期", ascending=False
    )
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "連結": st.column_config.LinkColumn("連結", display_text="開啟"),
            "開始日期": st.column_config.DateColumn("開始日期", format="YYYY-MM-DD"),
            "結束日期": st.column_config.DateColumn("結束日期", format="YYYY-MM-DD"),
            "公告日期": st.column_config.DateColumn("公告日期", format="YYYY-MM-DD"),
        },
    )
    st.download_button(
        "下載篩選後的 CSV",
        data=df.drop(columns=["開始月份"]).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ntpc_culture_{date.today():%Y%m%d}.csv",
        mime="text/csv",
    )

    with st.expander("查看單筆活動內容"):
        title = st.selectbox("選擇活動", show["標題"].tolist())
        row = df[df["標題"] == title].iloc[0]
        st.markdown(f"**{row['標題']}**（{row['類型']}）")
        st.write(row["內容"])
        if isinstance(row["連結"], str) and row["連結"]:
            st.link_button("前往原始網頁", row["連結"])
