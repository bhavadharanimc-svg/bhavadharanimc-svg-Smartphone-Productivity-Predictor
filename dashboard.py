import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="📱 Smartphone Usage Dashboard", page_icon="📱", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .section-title { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
        color: #ffffff; border-left: 4px solid #e94560; padding-left: 12px; margin: 2rem 0 1rem 0; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 12px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

COLORS = {"primary":"#e94560","teal":"#05c3c3","yellow":"#f5a623","accent":"#533483","bg":"#0d0d0d","card":"#1a1a2e","text":"#e2e8f0"}
PALETTE = [COLORS["primary"], COLORS["teal"], COLORS["yellow"], COLORS["accent"], "#6ee7b7", "#f87171"]
plt.rcParams.update({"figure.facecolor":COLORS["bg"],"axes.facecolor":COLORS["card"],"axes.edgecolor":"#2d3748",
    "axes.labelcolor":COLORS["text"],"xtick.color":COLORS["text"],"ytick.color":COLORS["text"],
    "text.color":COLORS["text"],"grid.color":"#2d3748","grid.linestyle":"--","grid.alpha":0.5})

@st.cache_data
def load_data():
    for path in ["smartphone_data.csv","data/smartphone_data.csv",
                 "Smartphone_Usage_Productivity_Dataset_50000.csv","data/Smartphone_Usage_Productivity_Dataset_50000.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Productivity_Category"] = pd.cut(df["Work_Productivity_Score"],bins=[0,3,6,10],labels=["Low","Medium","High"])
            df["Total_Screen_Time"] = df["Daily_Phone_Hours"] + df["Weekend_Screen_Time_Hours"]
            df["Social_Media_Ratio"] = (df["Social_Media_Hours"]/df["Daily_Phone_Hours"].replace(0,np.nan)).fillna(0)
            df["Age_Group"] = pd.cut(df["Age"],bins=[17,25,35,45,60],labels=["18-25","26-35","36-45","46-60"])
            return df
    st.error("❌ Dataset not found! Upload 'smartphone_data.csv' to your GitHub repo root."); st.stop()

df = load_data()

st.sidebar.markdown("## 🎛️ Filters")
sel_gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
sel_occ    = st.sidebar.multiselect("Occupation", df["Occupation"].unique(), default=list(df["Occupation"].unique()))
sel_device = st.sidebar.multiselect("Device Type", df["Device_Type"].unique(), default=list(df["Device_Type"].unique()))
age_range  = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 60))
f = df[df["Gender"].isin(sel_gender)&df["Occupation"].isin(sel_occ)&df["Device_Type"].isin(sel_device)&df["Age"].between(*age_range)]
st.sidebar.markdown(f"**Showing:** `{len(f):,}` of `{len(df):,}` records")

st.markdown("<h1 style='font-family:Syne;font-size:2.6rem;font-weight:800;color:#fff;margin-bottom:0'>📱 Smartphone Usage <span style='color:#e94560'>& Productivity</span></h1><p style='color:#a0aec0;margin-top:4px'>Interactive dashboard · 13 features analyzed</p>",unsafe_allow_html=True)
st.divider()

st.markdown('<div class="section-title">📊 Key Metrics</div>',unsafe_allow_html=True)
k1,k2,k3,k4,k5=st.columns(5)
k1.metric("👥 Total Users",f"{len(f):,}"); k2.metric("📱 Avg Phone Hours",f"{f['Daily_Phone_Hours'].mean():.1f} hrs")
k3.metric("😴 Avg Sleep",f"{f['Sleep_Hours'].mean():.1f} hrs"); k4.metric("⚡ Avg Productivity",f"{f['Work_Productivity_Score'].mean():.1f}/10")
k5.metric("😓 Avg Stress",f"{f['Stress_Level'].mean():.1f}/5")
st.divider()

st.markdown('<div class="section-title">🎯 Productivity Overview</div>',unsafe_allow_html=True)
c1,c2,c3=st.columns([1.2,1,1])
with c1:
    st.markdown("**Score Distribution**")
    fig,ax=plt.subplots(figsize=(6,3.5)); sc=f["Work_Productivity_Score"].value_counts().sort_index()
    ax.bar(sc.index,sc.values,color=COLORS["primary"],width=0.7,alpha=0.9); ax.bar(sc.idxmax(),sc.max(),color=COLORS["teal"],width=0.7)
    ax.set_xlabel("Score (1-10)"); ax.set_ylabel("Users"); ax.set_xticks(range(1,11)); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c2:
    st.markdown("**Category Split**")
    fig,ax=plt.subplots(figsize=(4,3.5)); cc=f["Productivity_Category"].value_counts()
    wedges,texts,autotexts=ax.pie(cc.values,labels=cc.index,autopct="%1.1f%%",colors=[COLORS["primary"],COLORS["yellow"],COLORS["teal"]],startangle=140,pctdistance=0.75,wedgeprops=dict(width=0.55,edgecolor=COLORS["bg"],linewidth=2))
    for t in autotexts: t.set_fontsize(10); t.set_color("white")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c3:
    st.markdown("**Avg Score by Gender**")
    fig,ax=plt.subplots(figsize=(4,3.5)); gp=f.groupby("Gender")["Work_Productivity_Score"].mean().sort_values()
    bars=ax.barh(gp.index,gp.values,color=PALETTE[:3],height=0.5)
    for bar,val in zip(bars,gp.values): ax.text(val+0.05,bar.get_y()+bar.get_height()/2,f"{val:.2f}",va="center",fontsize=10)
    ax.set_xlabel("Avg Score"); ax.set_xlim(0,11); ax.grid(axis="x")
    fig.tight_layout(); st.pyplot(fig); plt.close()
st.divider()

st.markdown('<div class="section-title">📱 Screen Time Analysis</div>',unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    st.markdown("**Daily Phone Hours by Occupation**")
    fig,ax=plt.subplots(figsize=(6,4)); occ_list=list(f["Occupation"].unique())
    occ_data=[f[f["Occupation"]==o]["Daily_Phone_Hours"].values for o in occ_list]
    bp=ax.boxplot(occ_data,labels=occ_list,patch_artist=True,medianprops=dict(color=COLORS["teal"],linewidth=2.5))
    for patch,color in zip(bp["boxes"],PALETTE): patch.set_facecolor(color); patch.set_alpha(0.7)
    for el in ["whiskers","fliers","caps"]:
        for item in bp[el]: item.set_color(COLORS["text"]); item.set_alpha(0.5)
    ax.set_ylabel("Daily Phone Hours"); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c2:
    st.markdown("**Social Media vs Productivity (colored by Stress)**")
    fig,ax=plt.subplots(figsize=(6,4)); sample=f.sample(min(500,len(f)),random_state=42)
    sc2=ax.scatter(sample["Social_Media_Hours"],sample["Work_Productivity_Score"],c=sample["Stress_Level"],cmap="RdYlGn_r",alpha=0.5,s=25,edgecolors="none")
    cbar=plt.colorbar(sc2,ax=ax); cbar.set_label("Stress Level")
    ax.set_xlabel("Social Media Hours/Day"); ax.set_ylabel("Productivity Score"); ax.grid(True)
    fig.tight_layout(); st.pyplot(fig); plt.close()
st.divider()

st.markdown('<div class="section-title">🏥 Health & Lifestyle</div>',unsafe_allow_html=True)
c1,c2,c3=st.columns(3)
with c1:
    st.markdown("**Sleep Hours Distribution**")
    fig,ax=plt.subplots(figsize=(4.5,3.5)); ax.hist(f["Sleep_Hours"],bins=20,color=COLORS["accent"],alpha=0.85,edgecolor=COLORS["bg"])
    ax.axvline(f["Sleep_Hours"].mean(),color=COLORS["teal"],linewidth=2,linestyle="--",label=f"Mean: {f['Sleep_Hours'].mean():.1f}h")
    ax.axvline(7,color=COLORS["yellow"],linewidth=1.5,linestyle=":",label="Recommended: 7h")
    ax.set_xlabel("Sleep Hours"); ax.set_ylabel("Count"); ax.legend(fontsize=8); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c2:
    st.markdown("**Stress by Occupation**")
    fig,ax=plt.subplots(figsize=(4.5,3.5)); so=f.groupby("Occupation")["Stress_Level"].mean().sort_values()
    colors_bar=[COLORS["teal"] if v<so.mean() else COLORS["primary"] for v in so.values]
    bars=ax.barh(so.index,so.values,color=colors_bar,height=0.5)
    ax.axvline(so.mean(),color=COLORS["yellow"],linewidth=1.5,linestyle="--",label="Average")
    for bar,val in zip(bars,so.values): ax.text(val+0.02,bar.get_y()+bar.get_height()/2,f"{val:.2f}",va="center",fontsize=9)
    ax.set_xlabel("Avg Stress Level"); ax.legend(fontsize=8); ax.grid(axis="x")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c3:
    st.markdown("**Caffeine vs Productivity**")
    fig,ax=plt.subplots(figsize=(4.5,3.5)); cp=f.groupby("Caffeine_Intake_Cups")["Work_Productivity_Score"].mean()
    ax.plot(cp.index,cp.values,color=COLORS["yellow"],linewidth=2.5,marker="o",markersize=6,markerfacecolor=COLORS["primary"])
    ax.fill_between(cp.index,cp.values,alpha=0.15,color=COLORS["yellow"])
    ax.set_xlabel("Caffeine Cups/Day"); ax.set_ylabel("Avg Productivity Score"); ax.grid(True)
    fig.tight_layout(); st.pyplot(fig); plt.close()
st.divider()

st.markdown('<div class="section-title">👥 Demographics</div>',unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    st.markdown("**Productivity by Age Group & Device**")
    fig,ax=plt.subplots(figsize=(6,4))
    ad=f.groupby(["Age_Group","Device_Type"],observed=True)["Work_Productivity_Score"].mean().unstack()
    x=np.arange(len(ad.index)); width=0.35
    for i,(col,color) in enumerate(zip(ad.columns,[COLORS["primary"],COLORS["teal"]])):
        ax.bar(x+i*width,ad[col],width,label=col,color=color,alpha=0.85)
    ax.set_xticks(x+width/2); ax.set_xticklabels(ad.index); ax.set_ylabel("Avg Productivity Score"); ax.legend(); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c2:
    st.markdown("**Heatmap: Productivity by Occupation × Stress**")
    fig,ax=plt.subplots(figsize=(6,4))
    hm=f.groupby(["Occupation","Stress_Level"])["Work_Productivity_Score"].mean().unstack()
    sns.heatmap(hm,ax=ax,cmap="RdYlGn",annot=True,fmt=".1f",linewidths=0.5,linecolor=COLORS["bg"],annot_kws={"size":9})
    ax.set_xlabel("Stress Level"); ax.set_ylabel("")
    fig.tight_layout(); st.pyplot(fig); plt.close()
st.divider()

st.markdown('<div class="section-title">🔗 Correlation & App Usage</div>',unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    st.markdown("**Correlation Heatmap**")
    fig,ax=plt.subplots(figsize=(6,5))
    num_cols=["Age","Daily_Phone_Hours","Social_Media_Hours","Work_Productivity_Score","Sleep_Hours","Stress_Level","App_Usage_Count","Caffeine_Intake_Cups","Weekend_Screen_Time_Hours"]
    corr=f[num_cols].corr(); mask=np.triu(np.ones_like(corr,dtype=bool))
    sns.heatmap(corr,mask=mask,ax=ax,cmap="coolwarm",center=0,annot=True,fmt=".2f",linewidths=0.5,linecolor=COLORS["bg"],annot_kws={"size":7.5},cbar_kws={"shrink":0.8})
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",fontsize=8); ax.set_yticklabels(ax.get_yticklabels(),fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()
with c2:
    st.markdown("**App Usage by Productivity Category**")
    fig,ax=plt.subplots(figsize=(6,5))
    for cat,color in zip(["Low","Medium","High"],[COLORS["primary"],COLORS["yellow"],COLORS["teal"]]):
        data=f[f["Productivity_Category"]==cat]["App_Usage_Count"]
        ax.hist(data,bins=20,alpha=0.6,label=cat,color=color,edgecolor="none")
    ax.set_xlabel("Apps Used Daily"); ax.set_ylabel("Users"); ax.legend(title="Productivity"); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()
st.divider()

with st.expander("🗃️ View Raw Data"):
    st.dataframe(f.reset_index(drop=True),use_container_width=True)
    st.download_button("⬇️ Download Filtered Dataset",data=f.to_csv(index=False),file_name="filtered_smartphone_data.csv",mime="text/csv")

st.caption("📱 Smartphone Usage & Productivity Dashboard · Built with Streamlit")
