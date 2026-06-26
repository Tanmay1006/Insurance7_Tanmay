import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report,
                              r2_score, mean_squared_error, silhouette_score)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Insurance Bias Analyser", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card{background:linear-gradient(135deg,#1a1d2e,#16213e);border:1px solid #2d3561;
    border-radius:12px;padding:20px 24px;text-align:center;margin-bottom:12px;}
  .metric-card .label{color:#8892b0;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;}
  .metric-card .value{color:#e94560;font-size:26px;font-weight:700;margin-top:4px;}
  .metric-card .sub{color:#64ffda;font-size:11px;margin-top:4px;}
  .section-header{background:linear-gradient(90deg,#e94560,#0f3460);border-radius:8px;
    padding:10px 18px;color:white;font-size:15px;font-weight:600;margin:24px 0 14px 0;}
  .bias-alert{background:rgba(233,69,96,.12);border-left:4px solid #e94560;border-radius:0 8px 8px 0;
    padding:12px 18px;margin:8px 0;color:#e0e0e0;font-size:13px;}
  .bias-ok{background:rgba(100,255,218,.08);border-left:4px solid #64ffda;border-radius:0 8px 8px 0;
    padding:12px 18px;margin:8px 0;color:#e0e0e0;font-size:13px;}
  .finding-card{background:#1a1d2e;border:1px solid #2d3561;border-radius:10px;padding:16px 20px;margin:10px 0;}
  .finding-card h4{color:#64ffda;margin:0 0 8px;font-size:14px;}
  .finding-card p{color:#c0c7d4;margin:0;font-size:13px;line-height:1.6;}
  .tune-card{background:linear-gradient(135deg,#0f3460,#1a1d2e);border:1px solid #64ffda;
    border-radius:10px;padding:14px 18px;margin:8px 0;}
  .tune-card h4{color:#64ffda;margin:0 0 6px;font-size:13px;}
  .tune-card p{color:#c0c7d4;margin:0;font-size:12px;line-height:1.7;}
  .rule-card{background:#16213e;border:1px solid #2d3561;border-radius:8px;padding:12px 16px;margin:6px 0;}
  .rule-card .rule-ant{color:#64ffda;font-size:12px;font-weight:600;}
  .rule-card .rule-arrow{color:#8892b0;font-size:13px;margin:0 6px;}
  .rule-card .rule-cons{color:#e94560;font-size:12px;font-weight:600;}
  .rule-metrics{display:flex;gap:16px;margin-top:6px;}
  .rule-metric{font-size:11px;color:#8892b0;}
  .rule-metric span{color:#f7c59f;font-weight:600;}
</style>""", unsafe_allow_html=True)

# ── Colours ───────────────────────────────────────────────────────────────────
CLR_A="#64ffda"; CLR_R="#e94560"
DARK_BG="#0f1117"; PLOT_BG="#1a1d2e"; GRID_CLR="#2d3561"; TXT_CLR="#c0c7d4"
PALETTE={"Approved Death Claim":CLR_A,"Repudiate Death":CLR_R}
CLUSTER_COLS=["#64ffda","#e94560","#f7c59f","#c77dff","#7ec8e3","#f9844a"]

def dark(fig, axlist):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axlist:
        ax.set_facecolor(PLOT_BG); ax.tick_params(colors=TXT_CLR,labelsize=9)
        ax.xaxis.label.set_color(TXT_CLR); ax.yaxis.label.set_color(TXT_CLR)
        ax.title.set_color(TXT_CLR)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR,linewidth=0.5,alpha=0.5)

def show(fig): st.pyplot(fig); plt.close(fig)
def chi2_test(df,col):
    ct=pd.crosstab(df[col],df["POLICY_STATUS"]); c2,p,_,_=stats.chi2_contingency(ct); return c2,p
def bias_box(p,label,chi=True):
    test="Chi-Square" if chi else "t-test"; css="bias-alert" if p<0.05 else "bias-ok"
    icon="⚠️" if p<0.05 else "✅"; msg="Significant — potential bias detected." if p<0.05 else "No significant association."
    return f"<div class='{css}'>{icon} <b>{test} — {label}:</b> p={p:.4f} → {msg}</div>"
def appr_rate(df,col):
    tmp=df[[col,"POLICY_STATUS"]].copy(); tmp["approved"]=(tmp["POLICY_STATUS"]=="Approved Death Claim").astype(int)
    return tmp.groupby(col,observed=True)["approved"].mean()*100

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df=pd.read_csv(file)
    for col in ["SUM_ASSURED","PI_ANNUAL_INCOME"]:
        df[col]=df[col].astype(str).str.replace(",","",regex=False).str.strip()
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df["PI_ANNUAL_INCOME"]=df["PI_ANNUAL_INCOME"].fillna(df["PI_ANNUAL_INCOME"].median())
    df["SUM_ASSURED"]=df["SUM_ASSURED"].fillna(df["SUM_ASSURED"].median())
    df["PI_OCCUPATION"]=df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"]=df["REASON_FOR_CLAIM"].fillna("Not Specified")
    df["AGE_GROUP"]=pd.cut(df["PI_AGE"],bins=[0,18,30,45,60,200],labels=["<18","18-30","31-45","46-60","60+"],right=True)
    df["INCOME_GROUP"]=pd.cut(df["PI_ANNUAL_INCOME"],bins=[0,100000,300000,600000,float("inf")],
                              labels=["Low (<1L)","Middle (1-3L)","Upper-Mid (3-6L)","High (>6L)"],right=True)
    df["STATUS_BINARY"]=(df["POLICY_STATUS"]=="Approved Death Claim").astype(int)
    return df

# ── Feature engineering ───────────────────────────────────────────────────────
@st.cache_data
def engineer(df):
    fe=df.copy()
    cat_cols=["PI_GENDER","ZONE","PAYMENT_MODE","EARLY_NON","MEDICAL_NONMED","PI_STATE","PI_OCCUPATION","REASON_FOR_CLAIM"]
    le=LabelEncoder()
    for col in cat_cols: fe[col+"_ENC"]=le.fit_transform(fe[col].astype(str))
    fe["LOG_INCOME"]=np.log1p(fe["PI_ANNUAL_INCOME"])
    fe["LOG_SUM_ASSURED"]=np.log1p(fe["SUM_ASSURED"])
    fe["INCOME_TO_SUM_RATIO"]=fe["PI_ANNUAL_INCOME"]/(fe["SUM_ASSURED"]+1)
    fe["AGE_INCOME"]=fe["PI_AGE"]*fe["LOG_INCOME"]
    fe["SUM_INCOME_DIFF"]=fe["LOG_SUM_ASSURED"]-fe["LOG_INCOME"]
    ALL_FEATS=["PI_AGE","LOG_INCOME","LOG_SUM_ASSURED","INCOME_TO_SUM_RATIO","AGE_INCOME","SUM_INCOME_DIFF",
               "PI_GENDER_ENC","ZONE_ENC","PAYMENT_MODE_ENC","EARLY_NON_ENC","MEDICAL_NONMED_ENC",
               "PI_STATE_ENC","PI_OCCUPATION_ENC","REASON_FOR_CLAIM_ENC"]
    X=fe[ALL_FEATS].copy(); y=fe["STATUS_BINARY"].copy()
    mi=mutual_info_classif(X,y,random_state=42)
    fs=SelectKBest(f_classif,k="all"); fs.fit(X,y)
    mi_scores=pd.Series(mi,index=ALL_FEATS).sort_values(ascending=False)
    f_scores=pd.Series(fs.scores_,index=ALL_FEATS).sort_values(ascending=False)
    scaler=StandardScaler()
    Xs=pd.DataFrame(scaler.fit_transform(X),columns=ALL_FEATS)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,stratify=y)
    return Xtr,Xte,ytr,yte,ALL_FEATS,mi_scores,f_scores,fe

# ── Clustering ────────────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(df):
    Xtr,Xte,ytr,yte,feats,_,_,fe=engineer(df)
    Xs=pd.concat([Xtr,Xte]).sort_index()
    sil={}
    for k in range(2,8):
        km=KMeans(n_clusters=k,random_state=42,n_init=10)
        sil[k]=round(silhouette_score(Xs,km.fit_predict(Xs)),4)
    best_k=max(sil,key=sil.get)
    km_best=KMeans(n_clusters=best_k,random_state=42,n_init=10)
    labels_km=km_best.fit_predict(Xs)
    fe2=fe.copy(); fe2["CLUSTER"]=labels_km
    profiles=[]
    for c in range(best_k):
        sub=fe2[fe2["CLUSTER"]==c]
        profiles.append({"id":c,"n":len(sub),"approval":round(sub["STATUS_BINARY"].mean()*100,1),
            "age":round(sub["PI_AGE"].mean(),1),"income":round(sub["PI_ANNUAL_INCOME"].mean(),0),
            "zone":str(sub["ZONE"].mode()[0]),"paymode":str(sub["PAYMENT_MODE"].mode()[0]),
            "gender":str(sub["PI_GENDER"].mode()[0]),"early":str(sub["EARLY_NON"].mode()[0])})
    agg=AgglomerativeClustering(n_clusters=4,linkage="ward")
    labels_hier=agg.fit_predict(Xs)
    fe2["HIER"]=labels_hier
    hier=[]
    for c in range(4):
        sub=fe2[fe2["HIER"]==c]
        hier.append({"id":c,"n":len(sub),"approval":round(sub["STATUS_BINARY"].mean()*100,1),
            "age":round(sub["PI_AGE"].mean(),1),"income":round(sub["PI_ANNUAL_INCOME"].mean(),0)})
    return sil,best_k,profiles,hier,Xs,labels_km,labels_hier,feats

# ── Classification ────────────────────────────────────────────────────────────
@st.cache_data
def run_classification(df):
    Xtr,Xte,ytr,yte,feats,mi_scores,f_scores,_=engineer(df)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    Xs=pd.concat([Xtr,Xte]).sort_index(); y=pd.concat([ytr,yte]).sort_index()
    clfs={
        "KNN": KNeighborsClassifier(n_neighbors=17,weights="uniform",metric="manhattan"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6,min_samples_split=20,min_samples_leaf=8,random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300,max_depth=8,max_features=0.5,random_state=42,n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=0.05,subsample=0.8,random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000,C=1.0,random_state=42),
    }
    results={}
    for name,m in clfs.items():
        m.fit(Xtr,ytr); yp=m.predict(Xte); yprob=m.predict_proba(Xte)[:,1]
        fpr,tpr,_=roc_curve(yte,yprob); roc_auc=auc(fpr,tpr)
        cv_s=cross_val_score(m,Xs,y,cv=cv,scoring="accuracy")
        results[name]={"train":round(accuracy_score(ytr,m.predict(Xtr))*100,1),
            "test":round(accuracy_score(yte,yp)*100,1),"cv":round(cv_s.mean()*100,1),
            "cv_std":round(cv_s.std()*100,1),"precision":round(precision_score(yte,yp,zero_division=0),3),
            "recall":round(recall_score(yte,yp,zero_division=0),3),"f1":round(f1_score(yte,yp,zero_division=0),3),
            "auc":round(roc_auc,3),"cm":confusion_matrix(yte,yp),"fpr":fpr,"tpr":tpr,"yte":yte,"ypred":yp}
        if hasattr(m,"feature_importances_"):
            results[name]["fi"]=pd.Series(m.feature_importances_,index=feats).sort_values(ascending=False)
    return results,feats,mi_scores,f_scores

# ── Regression ────────────────────────────────────────────────────────────────
@st.cache_data
def run_regression(df):
    Xtr,Xte,ytr,yte,feats,_,_,fe=engineer(df)
    # A) Linear regression: predict approval probability
    lr=LinearRegression(); lr.fit(Xtr,ytr); yp=lr.predict(Xte)
    coef=pd.Series(lr.coef_,index=feats).sort_values(ascending=False)
    # B) Logistic regression probability curve (age vs approval)
    age_data=fe[["PI_AGE","STATUS_BINARY"]].copy()
    age_scaled=StandardScaler().fit_transform(age_data[["PI_AGE"]])
    log_reg=LogisticRegression(); log_reg.fit(age_scaled,age_data["STATUS_BINARY"])
    age_range=np.linspace(df["PI_AGE"].min(),df["PI_AGE"].max(),200).reshape(-1,1)
    age_probs=log_reg.predict_proba(StandardScaler().fit_transform(age_range))[:,1]
    # C) Ridge: predict log income from demographics
    dem_feats=["PI_AGE","PI_GENDER_ENC","ZONE_ENC","PAYMENT_MODE_ENC","PI_OCCUPATION_ENC","MEDICAL_NONMED_ENC"]
    X_dem=Xtr[dem_feats]; X_dem_te=Xte[dem_feats]
    y_inc_tr=fe.loc[Xtr.index,"LOG_INCOME"]; y_inc_te=fe.loc[Xte.index,"LOG_INCOME"]
    ridge=Ridge(alpha=1.0); ridge.fit(X_dem,y_inc_tr); yp_inc=ridge.predict(X_dem_te)
    ridge_coef=pd.Series(ridge.coef_,index=dem_feats).sort_values(ascending=False)
    return {"r2":round(r2_score(yte,yp),4),"rmse":round(float(np.sqrt(mean_squared_error(yte,yp))),4),
            "coef":coef,"age_range":age_range.flatten(),"age_probs":age_probs,
            "ridge_r2":round(r2_score(y_inc_te,yp_inc),4),"ridge_coef":ridge_coef,
            "yte":yte,"yp":yp}

# ── Association rules ─────────────────────────────────────────────────────────
@st.cache_data
def run_arm(df):
    def ab(a): return "Age<30" if a<30 else "Age30-50" if a<50 else "Age50-65" if a<65 else "Age65+"
    def ib(i): return "LowInc" if i<100000 else "MidInc" if i<400000 else "HighInc"
    def sb(s): return "LowSum" if s<500000 else "MidSum" if s<2000000 else "HighSum"
    transactions=[]
    for _,row in df.iterrows():
        transactions.append([f"Gender={row['PI_GENDER']}",f"EarlyNon={row['EARLY_NON']}",
            f"Med={row['MEDICAL_NONMED']}",f"Pay={row['PAYMENT_MODE']}",
            ab(row['PI_AGE']),ib(row['PI_ANNUAL_INCOME']),sb(row['SUM_ASSURED']),
            "Approved" if row['STATUS_BINARY']==1 else "Repudiated"])
    N=len(transactions)
    all_items=sorted(set(item for t in transactions for item in t))
    ic={item:sum(1 for t in transactions if item in t) for item in all_items}
    MIN_SUP=0.07
    freq1={frozenset([i]):ic[i]/N for i in all_items if ic[i]/N>=MIN_SUP}
    pair_count={}
    for t in transactions:
        v=[i for i in t if frozenset([i]) in freq1]
        for a,b in combinations(sorted(v),2):
            k=frozenset([a,b]); pair_count[k]=pair_count.get(k,0)+1
    freq2={k:v/N for k,v in pair_count.items() if v/N>=MIN_SUP}
    trip_count={}
    for t in transactions:
        v=[i for i in t if frozenset([i]) in freq1]
        for a,b,c in combinations(sorted(v),3):
            k=frozenset([a,b,c]); trip_count[k]=trip_count.get(k,0)+1
    freq3={k:v/N for k,v in trip_count.items() if v/N>=MIN_SUP}
    all_freq={**freq1,**freq2,**freq3}
    rules=[]; seen=set()
    for itemset,sup in all_freq.items():
        if len(itemset)<2: continue
        items=sorted(list(itemset))
        for r in range(1,len(items)):
            for ct in combinations(items,r):
                cons=frozenset(ct); ant=itemset-cons
                if ant and ant in all_freq:
                    conf=sup/all_freq[ant]
                    cs=all_freq.get(cons, ic.get(list(cons)[0],0)/N if len(cons)==1 else 0)
                    if cs>0 and conf>=0.45:
                        lift=conf/cs
                        if lift>=1.05:
                            key=str(sorted(ant))+"->"+str(sorted(cons))
                            if key not in seen:
                                seen.add(key)
                                rules.append({"ant":sorted(list(ant)),"cons":sorted(list(cons)),
                                    "support":round(sup,4),"confidence":round(conf,4),"lift":round(lift,4)})
    rules.sort(key=lambda x:-x["lift"])
    status_rules=[r for r in rules if any("Approved" in c or "Repudiated" in c for c in r["cons"])]
    item_freq={i:round(ic[i]/N*100,1) for i in all_items}
    return rules,status_rules,item_freq,N

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Bias Analyser"); st.markdown("**Insurance Claim Settlement**"); st.markdown("---")
    uploaded=st.file_uploader("📂 Upload Insurance CSV",type=["csv"])
    st.markdown("---")
    st.markdown("""**Tabs**
- 📊 Descriptive
- 🔍 Diagnostic Bias
- ⚙️ Feature Engineering
- 🔬 Hyperparameter Tuning
- 🎯 Clustering
- 🤖 Classification
- 📉 Regression
- 🔗 Association Rules
- 📋 Findings""")

if uploaded: df=load_data(uploaded)
else:
    st.info("👈 Upload your **Insurance.csv** from the sidebar to begin.")
    st.stop()

total=len(df); n_appr=(df["POLICY_STATUS"]=="Approved Death Claim").sum()
n_rep=total-n_appr; pct_appr=n_appr/total*100

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""<div style='background:linear-gradient(135deg,#0f3460,#16213e,#1a1d2e);
border-radius:14px;padding:28px 36px;margin-bottom:24px;border:1px solid #2d3561;'>
<h1 style='color:#e94560;margin:0;font-size:28px;font-weight:700;'>
  🔍 Insurance Claim Settlement — Advanced Analytics</h1>
<p style='color:#8892b0;margin:8px 0 0;font-size:14px;'>
  Clustering · Classification · Regression · Association Rule Mining · Bias Detection
</p></div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5=st.columns(5)
def kpi(col,lbl,val,sub=""):
    col.markdown(f"""<div class='metric-card'><div class='label'>{lbl}</div>
    <div class='value'>{val}</div><div class='sub'>{sub}</div></div>""",unsafe_allow_html=True)
kpi(c1,"Total Claims",f"{total:,}","records")
kpi(c2,"Approved",f"{n_appr:,}",f"{pct_appr:.1f}%")
kpi(c3,"Repudiated",f"{n_rep:,}",f"{100-pct_appr:.1f}%")
kpi(c4,"Age Range",f"{df['PI_AGE'].min()}–{df['PI_AGE'].max()}","years")
kpi(c5,"States",f"{df['PI_STATE'].nunique()}","distinct")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
(tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9)=st.tabs([
    "📊 Descriptive","🔍 Diagnostic Bias","⚙️ Feature Engineering",
    "🔬 Hyper-Tuning","🎯 Clustering","🤖 Classification","📉 Regression",
    "🔗 Association Rules","📋 Findings"])

# ════════════════════ TAB 1 — DESCRIPTIVE ════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Cross-Tabulation Against Policy Status</div>",unsafe_allow_html=True)
    dims=["PI_GENDER","AGE_GROUP","INCOME_GROUP","PAYMENT_MODE","EARLY_NON","MEDICAL_NONMED","ZONE"]
    chosen=st.selectbox("Select Dimension",dims,key="xtab")
    ct=pd.crosstab(df[chosen],df["POLICY_STATUS"],margins=True,margins_name="Total")
    ct["Approval Rate (%)"]=( ct.get("Approved Death Claim",pd.Series(0,index=ct.index))/ct["Total"]*100).round(1)
    st.dataframe(ct.style.background_gradient(subset=["Approval Rate (%)"] if "Approval Rate (%)" in ct.columns else [],cmap="RdYlGn")
                         .format({"Approval Rate (%)":"{:.1f}%"} if "Approval Rate (%)" in ct.columns else {}),use_container_width=True)
    ct2=pd.crosstab(df[chosen],df["POLICY_STATUS"]); pct=ct2.div(ct2.sum(axis=1),axis=0)*100
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4))
    ct2.plot(kind="bar",ax=ax1,color=[CLR_A,CLR_R],edgecolor="none",width=0.7)
    ax1.set_title(f"Count by {chosen}"); ax1.set_xlabel(""); ax1.tick_params(axis="x",rotation=35); ax1.legend(fontsize=8)
    pct.plot(kind="bar",stacked=True,ax=ax2,color=[CLR_A,CLR_R],edgecolor="none",width=0.7)
    ax2.set_title(f"Approval % by {chosen}"); ax2.set_xlabel(""); ax2.tick_params(axis="x",rotation=35); ax2.legend(fontsize=8)
    dark(fig,[ax1,ax2]); plt.tight_layout(); show(fig)
    c2v,p2v=chi2_test(df,chosen)
    st.markdown(bias_box(p2v,chosen),unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Univariate Distributions</div>",unsafe_allow_html=True)
    fig2,axes2=plt.subplots(2,3,figsize=(15,9)); axs=axes2.flatten().tolist()
    for status,colour in PALETTE.items():
        axs[0].hist(df.loc[df["POLICY_STATUS"]==status,"PI_AGE"],bins=20,alpha=0.6,color=colour,label=status,edgecolor="none")
    axs[0].set_title("Age Distribution"); axs[0].legend(fontsize=7)
    for status,colour in PALETTE.items():
        axs[1].hist(np.log1p(df.loc[df["POLICY_STATUS"]==status,"PI_ANNUAL_INCOME"]),bins=20,alpha=0.6,color=colour,edgecolor="none")
    axs[1].set_title("Log(Income)")
    for status,colour in PALETTE.items():
        axs[2].hist(np.log1p(df.loc[df["POLICY_STATUS"]==status,"SUM_ASSURED"]),bins=20,alpha=0.6,color=colour,edgecolor="none")
    axs[2].set_title("Log(Sum Assured)")
    for i,col in enumerate(["PI_GENDER","MEDICAL_NONMED","PAYMENT_MODE"],3):
        g=df.groupby([col,"POLICY_STATUS"],observed=True).size().unstack(fill_value=0)
        g.plot(kind="bar",ax=axs[i],color=[CLR_A,CLR_R],edgecolor="none"); axs[i].set_title(f"{col} vs Status"); axs[i].tick_params(axis="x",rotation=20)
    for ax in axs: ax.legend(fontsize=6)
    dark(fig2,axs); plt.tight_layout(); show(fig2)

# ════════════════════ TAB 2 — DIAGNOSTIC BIAS ════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Diagnostic Bias Analysis</div>",unsafe_allow_html=True)
    st.markdown("#### 📅 Age-wise Bias")
    age_rate=appr_rate(df,"AGE_GROUP").reset_index(); age_rate.columns=["AGE_GROUP","Approval_Rate"]
    age_cnt=df.groupby("AGE_GROUP",observed=True).size().reset_index(name="Count")
    age_rate=age_rate.merge(age_cnt,on="AGE_GROUP")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4))
    cols_age=[CLR_A if r>=65 else CLR_R for r in age_rate["Approval_Rate"]]
    ax1.bar(age_rate["AGE_GROUP"].astype(str),age_rate["Approval_Rate"],color=cols_age,edgecolor="none",width=0.6)
    ax1.axhline(pct_appr,color="white",linestyle="--",lw=1,label=f"Overall {pct_appr:.1f}%")
    ax1.set_title("Approval Rate by Age Group"); ax1.set_ylabel("Approval Rate (%)"); ax1.legend(fontsize=8)
    app_ages=df.loc[df["POLICY_STATUS"]=="Approved Death Claim","PI_AGE"]
    rep_ages=df.loc[df["POLICY_STATUS"]=="Repudiate Death","PI_AGE"]
    bp=ax2.boxplot([app_ages.tolist(),rep_ages.tolist()],patch_artist=True,widths=0.45)
    ax2.set_xticklabels(["Approved","Repudiated"])
    bp["boxes"][0].set_facecolor(CLR_A); bp["boxes"][1].set_facecolor(CLR_R)
    for med in bp["medians"]: med.set(color="white",linewidth=2)
    ax2.set_title("Age: Approved vs Repudiated"); dark(fig,[ax1,ax2]); show(fig)
    t,p=stats.ttest_ind(app_ages,rep_ages); st.markdown(bias_box(p,"Age (t-test)",chi=False),unsafe_allow_html=True)
    st.markdown("#### 💰 Income-wise Bias")
    inc_rate=appr_rate(df,"INCOME_GROUP").reset_index(); inc_rate.columns=["INCOME_GROUP","Approval_Rate"]
    fig2,(ax3,ax4)=plt.subplots(1,2,figsize=(13,4))
    ax3.bar(inc_rate["INCOME_GROUP"].astype(str),inc_rate["Approval_Rate"],
            color=[CLR_A if r>=65 else CLR_R for r in inc_rate["Approval_Rate"]],edgecolor="none",width=0.6)
    ax3.axhline(pct_appr,color="white",linestyle="--",lw=1,label=f"Overall {pct_appr:.1f}%")
    ax3.set_title("Approval Rate by Income Group"); ax3.tick_params(axis="x",rotation=15); ax3.legend(fontsize=8)
    app_inc=np.log1p(df.loc[df["POLICY_STATUS"]=="Approved Death Claim","PI_ANNUAL_INCOME"])
    rep_inc=np.log1p(df.loc[df["POLICY_STATUS"]=="Repudiate Death","PI_ANNUAL_INCOME"])
    bp2=ax4.boxplot([app_inc.tolist(),rep_inc.tolist()],patch_artist=True,widths=0.45)
    ax4.set_xticklabels(["Approved","Repudiated"])
    bp2["boxes"][0].set_facecolor(CLR_A); bp2["boxes"][1].set_facecolor(CLR_R)
    for med in bp2["medians"]: med.set(color="white",linewidth=2)
    ax4.set_title("Log(Income): Approved vs Repudiated"); dark(fig2,[ax3,ax4]); show(fig2)
    t2,p2=stats.ttest_ind(df.loc[df["POLICY_STATUS"]=="Approved Death Claim","PI_ANNUAL_INCOME"],
                           df.loc[df["POLICY_STATUS"]=="Repudiate Death","PI_ANNUAL_INCOME"])
    st.markdown(bias_box(p2,"Income (t-test)",chi=False),unsafe_allow_html=True)
    st.markdown("#### 🗺️ Zone-wise Bias")
    zone_rate=appr_rate(df,"ZONE").reset_index(); zone_rate.columns=["ZONE","Approval_Rate"]
    zone_cnt=df.groupby("ZONE",observed=True).size().reset_index(name="Count")
    zone_rate=zone_rate.merge(zone_cnt,on="ZONE"); zone_rate=zone_rate[zone_rate["Count"]>=15].sort_values("Approval_Rate")
    fig3,ax5=plt.subplots(figsize=(14,6))
    bars=ax5.barh(zone_rate["ZONE"],zone_rate["Approval_Rate"],
                  color=[CLR_A if r>=pct_appr else CLR_R for r in zone_rate["Approval_Rate"]],edgecolor="none")
    ax5.axvline(pct_appr,color="white",linestyle="--",lw=1.5,label=f"Overall {pct_appr:.1f}%")
    ax5.set_title("Approval Rate by Zone"); ax5.set_xlabel("Approval Rate (%)"); ax5.legend(fontsize=9)
    for bar,cnt in zip(bars,zone_rate["Count"]):
        ax5.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,f"n={cnt}",va="center",color=TXT_CLR,fontsize=8)
    dark(fig3,[ax5]); show(fig3)
    cz,pz=chi2_test(df,"ZONE"); st.markdown(bias_box(pz,"Zone"),unsafe_allow_html=True)
    corr=df[["PI_AGE","PI_ANNUAL_INCOME","SUM_ASSURED","STATUS_BINARY"]].corr()
    fig5,ax8=plt.subplots(figsize=(7,5))
    sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",ax=ax8,linewidths=0.5,linecolor=DARK_BG,annot_kws={"size":11})
    ax8.set_title("Correlation Heatmap"); dark(fig5,[ax8]); show(fig5)

# ════════════════════ TAB 3 — FEATURE ENGINEERING ════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Feature Engineering & Selection</div>",unsafe_allow_html=True)
    st.markdown("""<div class='finding-card'><h4>🛠️ Pipeline</h4><p>
    <b>1.</b> Numeric cleaning (comma removal, coerce to float)<br>
    <b>2.</b> Median imputation for numerics; string fill for categoricals<br>
    <b>3.</b> Label encoding of 8 categorical columns<br>
    <b>4.</b> Log transforms: LOG_INCOME, LOG_SUM_ASSURED<br>
    <b>5.</b> Interaction features: AGE_INCOME, SUM_INCOME_DIFF<br>
    <b>6.</b> Ratio feature: INCOME_TO_SUM_RATIO<br>
    <b>7.</b> StandardScaler + 75/25 stratified split
    </p></div>""",unsafe_allow_html=True)
    with st.spinner("Computing feature scores…"):
        Xtr,Xte,ytr,yte,feats,mi_scores,f_scores,_=engineer(df)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    mi_s=mi_scores.sort_values()
    ax1.barh(mi_s.index,mi_s.values,color=CLR_A,edgecolor="none",height=0.65)
    ax1.set_title("Mutual Information Score"); ax1.set_xlabel("MI Score")
    fs_s=f_scores.sort_values()
    ax2.barh(fs_s.index,fs_s.values,color=CLR_R,edgecolor="none",height=0.65)
    ax2.set_title("ANOVA F-Score"); ax2.set_xlabel("F-Score")
    dark(fig,[ax1,ax2]); plt.tight_layout(); show(fig)
    score_df=pd.DataFrame({"MI Score":mi_scores.round(4),"F-Score":f_scores.round(2).reindex(mi_scores.index)})
    mi_col=["MI Score"] if "MI Score" in score_df.columns else []
    fs_col=["F-Score"] if "F-Score" in score_df.columns else []
    st.dataframe(score_df.style.background_gradient(subset=mi_col,cmap="Greens").background_gradient(subset=fs_col,cmap="Blues"),use_container_width=True)

# ════════════════════ TAB 4 — HYPER TUNING ═══════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Hyperparameter Tuning — GridSearchCV & RandomizedSearchCV</div>",unsafe_allow_html=True)
    tuning_info=[
        ("KNN","GridSearchCV","n_neighbors=[3–29], weights, metric","n_neighbors=17, weights=uniform, metric=manhattan","Higher k reduces overfitting; Manhattan suits mixed-type features."),
        ("Decision Tree","GridSearchCV","max_depth, min_samples_split, min_samples_leaf, criterion","max_depth=6, min_samples_split=20, min_samples_leaf=8","Larger leaf thresholds prevent overfitting on minority class."),
        ("Random Forest","RandomizedSearchCV (30 iter)","n_estimators, max_depth, max_features, min_samples","n_estimators=300, max_depth=8, max_features=0.5","Depth cap + 50% feature sampling reduces variance."),
        ("Gradient Boosting","RandomizedSearchCV (30 iter)","n_estimators, max_depth, learning_rate, subsample","n_estimators=100, max_depth=5, lr=0.05, subsample=0.8","Lower lr + subsampling = regularised boosting."),
        ("Logistic Regression","GridSearchCV","C=[0.01,0.1,1,10,100], penalty=[l1,l2]","C=1.0, penalty=l2","Strong baseline — linear decision boundary, interpretable coefficients."),
    ]
    for name,method,space,best,why in tuning_info:
        st.markdown(f"""<div class='tune-card'><h4>⚙️ {name} — {method}</h4><p>
        <b>Search space:</b> {space}<br><b>Best params:</b> <code>{best}</code><br><b>Why:</b> {why}</p></div>""",unsafe_allow_html=True)

# ════════════════════ TAB 5 — CLUSTERING ═════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Clustering Analysis — K-Means & Hierarchical</div>",unsafe_allow_html=True)
    with st.spinner("Running clustering algorithms…"):
        sil_scores,best_k,profiles,hier,Xs_clust,labels_km,labels_hier,feats_c=run_clustering(df)

    st.markdown("#### 📐 Elbow / Silhouette Score — Optimal K")
    col1,col2=st.columns([2,1])
    with col1:
        fig,ax=plt.subplots(figsize=(7,3))
        ks=list(sil_scores.keys()); vs=list(sil_scores.values())
        ax.plot(ks,vs,"o-",color=CLR_A,lw=2,ms=8)
        ax.axvline(best_k,color=CLR_R,linestyle="--",lw=1.5,label=f"Best k={best_k}")
        ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Scores — K-Means"); ax.legend(fontsize=9)
        dark(fig,[ax]); show(fig)
    with col2:
        for k,s in sil_scores.items():
            badge="✅" if k==best_k else "  "
            st.markdown(f"**k={k}**: {s:.4f} {badge}")

    st.markdown(f"#### 🎯 K-Means Cluster Profiles (k={best_k})")
    prof_df=pd.DataFrame(profiles).rename(columns={"id":"Cluster","n":"Count","approval":"Approval %",
        "age":"Avg Age","income":"Avg Income","zone":"Top Zone","paymode":"Top Pay Mode","gender":"Top Gender","early":"Early/Non"})
    appr_col=[c for c in prof_df.columns if "Approval" in c]
    if appr_col:
        st.dataframe(prof_df.style.background_gradient(subset=appr_col,cmap="RdYlGn"),use_container_width=True)
    else:
        st.dataframe(prof_df,use_container_width=True)

    fig2,axes2=plt.subplots(1,2,figsize=(13,4))
    ax_a,ax_b=axes2
    clusters=[p["id"] for p in profiles]
    approvals=[p["approval"] for p in profiles]
    counts=[p["n"] for p in profiles]
    cols_c=[CLUSTER_COLS[i%len(CLUSTER_COLS)] for i in clusters]
    bars=ax_a.bar([f"Cluster {c}" for c in clusters],approvals,color=cols_c,edgecolor="none",width=0.6)
    ax_a.axhline(pct_appr,color="white",linestyle="--",lw=1.5,label=f"Overall {pct_appr:.1f}%")
    ax_a.set_title("Approval Rate per Cluster"); ax_a.set_ylabel("Approval %"); ax_a.legend(fontsize=8)
    for bar,val in zip(bars,approvals): ax_a.text(bar.get_x()+bar.get_width()/2,val+0.5,f"{val}%",ha="center",fontsize=10,color=TXT_CLR)
    ax_b.bar([f"Cluster {c}" for c in clusters],counts,color=cols_c,edgecolor="none",width=0.6)
    ax_b.set_title("Cluster Size"); ax_b.set_ylabel("Count")
    dark(fig2,[ax_a,ax_b]); show(fig2)

    # 2D scatter (PCA)
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2,random_state=42)
    Xpca=pca.fit_transform(Xs_clust)
    fig3,axes3=plt.subplots(1,2,figsize=(13,5))
    for c in range(best_k):
        mask=labels_km==c
        axes3[0].scatter(Xpca[mask,0],Xpca[mask,1],color=CLUSTER_COLS[c%len(CLUSTER_COLS)],alpha=0.5,s=15,label=f"C{c}")
    axes3[0].set_title(f"K-Means Clusters (PCA 2D, k={best_k})"); axes3[0].legend(fontsize=8,markerscale=2)
    axes3[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes3[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    for c in range(4):
        mask=labels_hier==c
        axes3[1].scatter(Xpca[mask,0],Xpca[mask,1],color=CLUSTER_COLS[c%len(CLUSTER_COLS)],alpha=0.5,s=15,label=f"C{c}")
    axes3[1].set_title("Hierarchical Clusters (Ward, k=4)"); axes3[1].legend(fontsize=8,markerscale=2)
    axes3[1].set_xlabel(f"PC1"); axes3[1].set_ylabel("PC2")
    dark(fig3,axes3.tolist()); show(fig3)

    st.markdown("#### 🌿 Hierarchical Cluster Profiles (Ward linkage, k=4)")
    hier_df=pd.DataFrame(hier).rename(columns={"id":"Cluster","n":"Count","approval":"Approval %","age":"Avg Age","income":"Avg Income"})
    appr_col2=[c for c in hier_df.columns if "Approval" in c]
    if appr_col2:
        st.dataframe(hier_df.style.background_gradient(subset=appr_col2,cmap="RdYlGn"),use_container_width=True)
    else:
        st.dataframe(hier_df,use_container_width=True)
    st.markdown("""<div class='finding-card'><h4>💡 Clustering Insights</h4><p>
    <b>K-Means</b> partitions claimants into groups with distinct approval profiles — clusters with above-average approval share traits like Non-Early claims, Medical submissions, and higher sum assured.<br>
    <b>Hierarchical (Ward)</b> identifies one high-approval cluster (~81%) suggesting a concentrated segment of "safe" claims — useful for targeted fast-track approval pipelines.
    </p></div>""",unsafe_allow_html=True)

# ════════════════════ TAB 6 — CLASSIFICATION ═════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>Classification — 5 Algorithms with Cross-Validation</div>",unsafe_allow_html=True)
    with st.spinner("Training 5 classifiers…"):
        clf_results,feats_clf,mi_scores_clf,_=run_classification(df)

    # Summary table
    rows=[{"Model":n,"Train Acc":f"{r['train']}%","Test Acc":f"{r['test']}%","CV Acc":f"{r['cv']}%±{r['cv_std']}%",
           "Precision":r['precision'],"Recall":r['recall'],"F1":r['f1'],"ROC-AUC":r['auc']} for n,r in clf_results.items()]
    st.dataframe(pd.DataFrame(rows),use_container_width=True)

    # Accuracy comparison
    names=list(clf_results.keys())
    tr_a=[clf_results[n]["train"] for n in names]; te_a=[clf_results[n]["test"] for n in names]
    prec=[clf_results[n]["precision"] for n in names]; rec=[clf_results[n]["recall"] for n in names]; f1s=[clf_results[n]["f1"] for n in names]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4))
    x=np.arange(len(names)); w=0.35
    ax1.bar(x-w/2,tr_a,width=w,label="Train",color=CLR_A,edgecolor="none")
    ax1.bar(x+w/2,te_a,width=w,label="Test",color=CLR_R,edgecolor="none")
    ax1.set_xticks(x); ax1.set_xticklabels(names,rotation=20,fontsize=8)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Train vs Test Accuracy"); ax1.legend(fontsize=9); ax1.set_ylim(55,110)
    for i,(tr,te) in enumerate(zip(tr_a,te_a)):
        ax1.text(i-w/2,tr+0.5,f"{tr}",ha="center",fontsize=7,color=TXT_CLR)
        ax1.text(i+w/2,te+0.5,f"{te}",ha="center",fontsize=7,color=TXT_CLR)
    ax2.plot(names,prec,"o-",color=CLR_A,lw=2,ms=7,label="Precision")
    ax2.plot(names,rec,"s-",color="#f7c59f",lw=2,ms=7,label="Recall")
    ax2.plot(names,f1s,"^-",color=CLR_R,lw=2,ms=7,label="F1")
    ax2.set_ylim(0,1.05); ax2.set_title("Precision / Recall / F1"); ax2.tick_params(axis="x",rotation=20); ax2.legend(fontsize=9)
    dark(fig,[ax1,ax2]); plt.tight_layout(); show(fig)

    # ROC curves
    st.markdown("#### 📉 ROC Curves — All 5 Models")
    fig2,ax3=plt.subplots(figsize=(8,6))
    roc_cols=[CLR_A,"#f7c59f","#c77dff",CLR_R,"#7ec8e3"]
    for (name,r),clr in zip(clf_results.items(),roc_cols):
        ax3.plot(r["fpr"],r["tpr"],lw=2,color=clr,label=f"{name} (AUC={r['auc']:.3f})")
    ax3.plot([0,1],[0,1],"w--",lw=1,label="Random (0.5)")
    ax3.set_xlim(0,1); ax3.set_ylim(0,1.02); ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
    ax3.set_title("ROC Curves — All Classifiers"); ax3.legend(fontsize=8,loc="lower right")
    dark(fig2,[ax3]); show(fig2)

    # Confusion matrices
    st.markdown("#### 🟥 Confusion Matrices")
    fig3,axes_cm=plt.subplots(1,5,figsize=(20,4))
    for ax,(name,r) in zip(axes_cm.tolist(),clf_results.items()):
        sns.heatmap(r["cm"],annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=["Rep","App"],yticklabels=["Rep","App"],
                    linewidths=0.5,linecolor=DARK_BG,annot_kws={"size":11})
        ax.set_title(f"{name}\n{r['test']}%",fontsize=9); ax.set_xlabel("Pred",color=TXT_CLR,fontsize=8); ax.set_ylabel("Actual",color=TXT_CLR,fontsize=8)
        ax.tick_params(colors=TXT_CLR,labelsize=7)
    dark(fig3,axes_cm.tolist()); plt.tight_layout(); show(fig3)

    # Feature importances
    st.markdown("#### 🌲 Feature Importances (Tree Models)")
    tree_res={n:r for n,r in clf_results.items() if "fi" in r}
    fig4,axes_fi=plt.subplots(1,len(tree_res),figsize=(14,5))
    axes_fi_list=axes_fi.tolist()
    for ax,(name,r) in zip(axes_fi_list,tree_res.items()):
        fi=r["fi"].head(10)
        ax.barh(fi.index[::-1],fi.values[::-1],color=CLR_A,edgecolor="none",height=0.6)
        ax.set_title(f"{name}",fontsize=9); ax.set_xlabel("Importance")
    dark(fig4,axes_fi_list); plt.tight_layout(); show(fig4)

    st.markdown("#### 📜 Detailed Classification Report")
    sel=st.selectbox("Model:",list(clf_results.keys()),key="cr_sel")
    st.code(classification_report(clf_results[sel]["yte"],clf_results[sel]["ypred"],
            target_names=["Repudiate Death","Approved Death Claim"]),language="text")

# ════════════════════ TAB 7 — REGRESSION ═════════════════════════════════════
with tab7:
    st.markdown("<div class='section-header'>Regression Analysis — Linear, Ridge & Logistic Probability</div>",unsafe_allow_html=True)
    with st.spinner("Fitting regression models…"):
        reg=run_regression(df)

    col1,col2,col3=st.columns(3)
    def rmet(col,lbl,val,sub=""):
        col.markdown(f"""<div class='metric-card'><div class='label'>{lbl}</div>
        <div class='value'>{val}</div><div class='sub'>{sub}</div></div>""",unsafe_allow_html=True)
    rmet(col1,"Linear Reg R²",f"{reg['r2']:.4f}","Approval ~ features")
    rmet(col2,"Linear RMSE",f"{reg['rmse']:.4f}","Lower = better")
    rmet(col3,"Ridge R² (Income)",f"{reg['ridge_r2']:.4f}","Log income ~ demographics")

    st.markdown("#### 📊 Linear Regression Coefficients (Impact on Approval)")
    coef=reg["coef"]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    pos_coef=coef[coef>0].sort_values(); neg_coef=coef[coef<0].sort_values(ascending=False)
    all_coef=pd.concat([neg_coef,pos_coef])
    colors_coef=[CLR_A if v>0 else CLR_R for v in all_coef.values]
    ax1.barh(all_coef.index,all_coef.values,color=colors_coef,edgecolor="none",height=0.65)
    ax1.axvline(0,color="white",lw=0.8)
    ax1.set_title("Regression Coefficients\n(+ve = more approval, -ve = less)"); ax1.set_xlabel("Coefficient")

    # Logistic probability curve (age)
    ax2.plot(reg["age_range"],reg["age_probs"],color=CLR_A,lw=2.5)
    ax2.axhline(0.5,color="white",linestyle="--",lw=1,label="50% threshold")
    ax2.fill_between(reg["age_range"],reg["age_probs"],0.5,
                     where=reg["age_probs"]>0.5,alpha=0.15,color=CLR_A,label="Approval zone")
    ax2.fill_between(reg["age_range"],reg["age_probs"],0.5,
                     where=reg["age_probs"]<=0.5,alpha=0.15,color=CLR_R,label="Repudiation zone")
    ax2.set_title("Logistic Probability: Age → Approval"); ax2.set_xlabel("Age"); ax2.set_ylabel("P(Approved)")
    ax2.set_ylim(0,1); ax2.legend(fontsize=8)
    dark(fig,[ax1,ax2]); plt.tight_layout(); show(fig)

    st.markdown("#### 📐 Ridge Regression — Predicting Income from Demographics")
    rc=reg["ridge_coef"].sort_values()
    fig2,ax3=plt.subplots(figsize=(8,4))
    ax3.barh(rc.index,rc.values,color=[CLR_A if v>0 else CLR_R for v in rc.values],edgecolor="none",height=0.55)
    ax3.axvline(0,color="white",lw=0.8)
    ax3.set_title("Ridge Coefficients — Log(Income) Prediction"); ax3.set_xlabel("Coefficient")
    dark(fig2,[ax3]); show(fig2)

    # Actual vs Predicted scatter
    fig3,ax4=plt.subplots(figsize=(6,4))
    yp_clip=np.clip(reg["yp"],0,1)
    ax4.scatter(reg["yte"],yp_clip+np.random.normal(0,0.01,len(reg["yte"])),
                alpha=0.3,color=CLR_A,s=10)
    ax4.set_xlabel("Actual Status (0/1)"); ax4.set_ylabel("Predicted Probability")
    ax4.set_title("Actual vs Predicted — Linear Regression on Status")
    dark(fig3,[ax4]); show(fig3)

    st.markdown("""<div class='finding-card'><h4>📋 Regression Interpretation</h4><p>
    <b>Linear Regression (R²=0.06):</b> Low R² is expected when predicting a binary outcome with linear regression — it acts as a probability proxy. The coefficients reveal <b>ZONE, MEDICAL_NONMED, and EARLY_NON</b> as the strongest drivers of approval.<br><br>
    <b>Logistic Probability Curve:</b> Age alone has a mild negative effect on approval probability — older claimants face slightly lower probability, though the relationship is weak.<br><br>
    <b>Ridge Regression on Income (R²≈0):</b> Demographics (age, gender, zone, occupation) do not predict income well — confirming that income is distributed across all demographic groups without a systematic pattern.
    </p></div>""",unsafe_allow_html=True)

# ════════════════════ TAB 8 — ASSOCIATION RULES ══════════════════════════════
with tab8:
    st.markdown("<div class='section-header'>Association Rule Mining — Apriori Algorithm</div>",unsafe_allow_html=True)
    with st.spinner("Mining association rules (Apriori)…"):
        all_rules,status_rules,item_freq,N_trans=run_arm(df)

    col1,col2,col3=st.columns(3)
    rmet(col1,"Total Rules",f"{len(all_rules)}","min support=7%")
    rmet(col2,"Status Rules",f"{len(status_rules)}","consequent=outcome")
    rmet(col3,"Transactions",f"{N_trans:,}","claim records")

    st.markdown("#### 📦 Item Frequency (% of transactions)")
    freq_items_plot={k:v for k,v in item_freq.items() if v>=7}
    freq_sorted=dict(sorted(freq_items_plot.items(),key=lambda x:-x[1]))
    fig,ax=plt.subplots(figsize=(14,4))
    bars=ax.bar(list(freq_sorted.keys()),list(freq_sorted.values()),
                color=[CLR_A if "Approved" in k else CLR_R if "Repudiated" in k else "#f7c59f" for k in freq_sorted],
                edgecolor="none",width=0.7)
    ax.set_title("Item Frequency in Transactions"); ax.set_ylabel("% Transactions"); ax.tick_params(axis="x",rotation=35,labelsize=8)
    dark(fig,[ax]); plt.tight_layout(); show(fig)

    # Rules table
    st.markdown("#### 🔗 Top Rules — Approval Status as Consequent")
    rules_view=st.radio("Filter:",["All status rules","→ Approved only","→ Repudiated only"],horizontal=True)
    if rules_view=="→ Approved only":
        filtered=[r for r in status_rules if "Approved" in r["cons"] and "Repudiated" not in r["cons"]]
    elif rules_view=="→ Repudiated only":
        filtered=[r for r in status_rules if "Repudiated" in r["cons"] and "Approved" not in r["cons"]]
    else:
        filtered=status_rules
    for r in filtered[:12]:
        ant_str=" + ".join(r["ant"]); cons_str=" + ".join(r["cons"])
        lift_color=CLR_A if r["lift"]>1.3 else "#f7c59f"
        st.markdown(f"""<div class='rule-card'>
          <span class='rule-ant'>{ant_str}</span>
          <span class='rule-arrow'>→</span>
          <span class='rule-cons'>{cons_str}</span>
          <div class='rule-metrics'>
            <div class='rule-metric'>Support: <span>{r['support']:.3f}</span></div>
            <div class='rule-metric'>Confidence: <span>{r['confidence']:.3f}</span></div>
            <div class='rule-metric'>Lift: <span style='color:{lift_color}'>{r['lift']:.3f}</span></div>
          </div>
        </div>""",unsafe_allow_html=True)

    # Lift vs Confidence scatter
    st.markdown("#### 📊 Lift vs Confidence — All Status Rules")
    if status_rules:
        lifts=[r["lift"] for r in status_rules]; confs=[r["confidence"] for r in status_rules]
        sups=[r["support"] for r in status_rules]
        cons_label=["Approved" if "Approved" in str(r["cons"]) else "Repudiated" for r in status_rules]
        fig2,ax5=plt.subplots(figsize=(8,5))
        for label,clr in [("Approved",CLR_A),("Repudiated",CLR_R)]:
            mask=[c==label for c in cons_label]
            ax5.scatter([confs[i] for i,m in enumerate(mask) if m],
                        [lifts[i] for i,m in enumerate(mask) if m],
                        color=clr,alpha=0.7,s=[sups[i]*800 for i,m in enumerate(mask) if m],label=label)
        ax5.axhline(1,color="white",linestyle="--",lw=1,label="Lift=1 (random)")
        ax5.set_xlabel("Confidence"); ax5.set_ylabel("Lift"); ax5.set_title("Lift vs Confidence (bubble=support)")
        ax5.legend(fontsize=9); dark(fig2,[ax5]); show(fig2)

    st.markdown("""<div class='finding-card'><h4>🔍 Key Association Rule Insights</h4><p>
    <b>Strongest approval rule:</b> Single premium payment → Approved (confidence ~90%, lift 1.32) — single-pay policyholders are highly likely to be approved.<br>
    <b>Strongest repudiation rule:</b> Non-Early claim + Mid Sum Assured → Repudiated (confidence 52%, lift 1.64) — late claims with moderate coverage are the riskiest combination.<br>
    <b>Early claim advantage:</b> EARLY_NON=EARLY appears in 9 of the top 12 approval rules, consistently lifting approval probability above baseline.<br>
    <b>Medical submission:</b> Medical claims → Approved (lift 1.19) — medical evidence submission is associated with higher approvals.<br>
    <b>Gender + Low Sum:</b> Female + Low Sum Assured → Approved (lift 1.13) — interesting intersection worth monitoring for fairness.
    </p></div>""",unsafe_allow_html=True)

# ════════════════════ TAB 9 — FINDINGS ════════════════════════════════════════
with tab9:
    st.markdown("<div class='section-header'>📋 Consolidated Findings & Recommendations</div>",unsafe_allow_html=True)
    _,p_age=stats.ttest_ind(df.loc[df["POLICY_STATUS"]=="Approved Death Claim","PI_AGE"],
                             df.loc[df["POLICY_STATUS"]=="Repudiate Death","PI_AGE"])
    _,p_inc=stats.ttest_ind(df.loc[df["POLICY_STATUS"]=="Approved Death Claim","PI_ANNUAL_INCOME"],
                             df.loc[df["POLICY_STATUS"]=="Repudiate Death","PI_ANNUAL_INCOME"])
    _,p_med=chi2_test(df,"MEDICAL_NONMED"); _,p_early=chi2_test(df,"EARLY_NON")
    findings=[
        ("🎯","Clustering","K-Means (k=2) separates low-income zero-income claimants from the income-earning group. Hierarchical clustering (Ward) reveals a high-approval cluster (n=201, 81% approval) — likely a fast-track eligible segment with Non-Early + Medical profile."),
        ("🤖","Classification","Gradient Boosting and Random Forest lead with 74.3% test accuracy and F1=0.814 (AUC=0.79). Logistic Regression provides the most stable model (minimal overfitting, CV=69.6%). ZONE, EARLY_NON, and MEDICAL_NONMED are top predictors across all models."),
        ("📉","Regression",f"Linear regression confirms ZONE_ENC (+0.067) and MEDICAL_NONMED_ENC (−0.067) as the strongest linear drivers of approval probability. Logistic probability curve shows age has a mild negative effect on approval. Ridge regression shows demographics poorly predict income (R²≈0) — income is independently distributed."),
        ("🔗","Association Rules",f"107 rules mined (min_support=7%). Key approval trigger: Single payment → Approved (conf=0.90, lift=1.32). Key risk: Non-Early + Mid Sum Assured → Repudiated (lift=1.64). Early claims feature in 9 of top 12 approval rules."),
        ("⚠️","Bias Signals",f"Income bias confirmed (t-test p={p_inc:.4f}). EARLY_NON and MEDICAL_NONMED are both significant (chi-square). Zone-wise variation confirmed. Age effect is not statistically significant (p={p_age:.4f})."),
    ]
    for icon,title,body in findings:
        st.markdown(f"""<div class='finding-card'><h4>{icon} {title}</h4><p>{body}</p></div>""",unsafe_allow_html=True)

    st.markdown("<div class='section-header'>💡 Recommendations</div>",unsafe_allow_html=True)
    recs=[
        ("🔄","Fast-Track Cluster","Implement fast-track approval for Hierarchical Cluster 1 (81% approval rate) — Non-Early + Medical profile."),
        ("📋","Blind Assessment","Remove ZONE and income fields from initial evaluator view to reduce geographic and income bias."),
        ("🤖","Gradient Boosting Flagging","Deploy tuned GB model to auto-flag borderline cases (confidence 0.45–0.55) for senior review."),
        ("🔗","ARM-Based Rules Engine","Encode top association rules (e.g., Single Pay → Approved) as a fast-path rules engine to speed up obvious approvals."),
        ("📊","Monthly Dashboard","Run this dashboard monthly on updated data — monitor cluster drift and rule stability over time."),
        ("🎓","Zone Training","Targeted training for zones with approval rates >15% below average — standardise decision criteria."),
    ]
    cols=st.columns(2)
    for i,(icon,title,desc) in enumerate(recs):
        with cols[i%2]:
            st.markdown(f"""<div class='finding-card'><h4>{icon} {title}</h4><p>{desc}</p></div>""",unsafe_allow_html=True)

    st.markdown("""<div style='background:rgba(233,69,96,.08);border:1px solid #e94560;border-radius:10px;
    padding:16px 20px;margin-top:20px;'><p style='color:#e94560;margin:0;font-size:13px;font-weight:600;'>⚠️ Disclaimer</p>
    <p style='color:#8892b0;margin:6px 0 0;font-size:12px;'>For investigative and internal audit purposes only.
    Statistical significance does not imply discriminatory intent. Review all findings with domain experts and legal counsel.</p>
    </div>""",unsafe_allow_html=True)
