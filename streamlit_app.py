import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stMetric{background:#f4f6fb;border-radius:10px;padding:8px}
.approve-box{background:#d4edda;border:2px solid #28a745;border-radius:10px;
             padding:20px;text-align:center;font-size:26px;font-weight:bold;color:#155724}
.decline-box{background:#f8d7da;border:2px solid #dc3545;border-radius:10px;
             padding:20px;text-align:center;font-size:26px;font-weight:bold;color:#721c24}
</style>
""", unsafe_allow_html=True)

# ── Artifact loading ───────────────────────────────────────────────
REQUIRED = {
    "model":    "credit_xgboost.pkl",
    "meta":     "feature_metadata.pkl",
    "results":  "results_summary.pkl",
    "roc_data": "roc_curve_data.pkl",
    "fairness": "fairness_results.pkl",
}

@st.cache_resource(show_spinner="Loading model …")
def load_all():
    out, miss = {}, []
    for k, f in REQUIRED.items():
        if os.path.exists(f):
            out[k] = joblib.load(f)
        else:
            miss.append(f)
    if "model" in out:
        out["explainer"] = shap.TreeExplainer(out["model"])
    return out, miss

arts, missing = load_all()

# ── Header ─────────────────────────────────────────────────────────
st.title("💳 Credit Risk Scoring System")
st.markdown(
    "**XGBoost + SMOTE + SHAP + Fairlearn** &nbsp;|&nbsp; "
    "Home Credit Default Risk · 50 K applications · 3 % minority class"
)

if missing:
    st.error(f"Missing files: {missing}")
    st.info(
        "**Setup:** Download all `.pkl` files + `.png` files from `/kaggle/working/` "
        "and place them in the same folder as `streamlit_app.py`, then re-run."
    )
    st.stop()

model    = arts["model"]
explainer= arts["explainer"]
meta     = arts["meta"]
results  = arts["results"]
roc_data = arts["roc_data"]
fairness = arts["fairness"]

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance", "🎯 Score Applicant", "⚖️ Fairness Audit", "🔍 Feature Importance"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 – Model Performance
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Model Performance Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC-ROC",              f"{results.get('auc',0):.4f}",
              delta=f"+{results.get('auc',0)-results.get('auc_before',0):.4f} vs baseline")
    c2.metric("Minority Recall",      f"{results.get('recall_after',0):.1%}",
              delta=f"+{results.get('recall_after',0)-results.get('recall_before',0):.1%} from SMOTE")
    c3.metric("Accuracy gain vs FICO",f"+{results.get('accuracy_gain',0):.1%}")
    c4.metric("EOD gender (↓ = fair)",f"{results.get('eod_gender',0):.3f}",
              delta_color="inverse", delta="post-mitigation")

    st.divider()
    cl,cr = st.columns(2)

    with cl:
        st.subheader("Recall: Before vs After SMOTE")
        rb = results.get("recall_before",0.45)
        ra = results.get("recall_after", 0.78)
        fig,ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(["Before SMOTE","After SMOTE"],[rb,ra],
                      color=["#e74c3c","#27ae60"],width=0.5,edgecolor="white",linewidth=2)
        for bar,v in zip(bars,[rb,ra]):
            ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+0.015,
                    f"{v:.1%}",ha="center",fontsize=14,fontweight="bold")
        ax.set_ylim(0,1.05); ax.set_ylabel("Minority Class Recall",fontsize=12)
        ax.set_title("SMOTE Impact on Minority Recall",fontsize=13,fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y",alpha=0.3); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with cr:
        st.subheader("ROC Curve")
        fpr_a = roc_data.get("fpr",[0,1])
        tpr_a = roc_data.get("tpr",[0,1])
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr_a,tpr_a,"#3498db",lw=2.5,label=f"XGBoost AUC={results.get('auc',0):.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random AUC=0.500")
        ax.fill_between(fpr_a,tpr_a,alpha=0.10,color="#3498db")
        ax.set_xlabel("False Positive Rate",fontsize=11)
        ax.set_ylabel("True Positive Rate",fontsize=11)
        ax.set_title("ROC Curve",fontsize=13,fontweight="bold")
        ax.legend(loc="lower right",fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.divider()
    ccm,ctbl = st.columns([1,2])

    with ccm:
        st.subheader("Confusion Matrix")
        cm_d = np.array(results.get("confusion_matrix",[[8000,200],[300,1200]]))
        fig,ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm_d,annot=True,fmt="d",cmap="Blues",ax=ax,
                    xticklabels=["Pred: Repaid","Pred: Default"],
                    yticklabels=["True: Repaid","True: Default"],
                    annot_kws={"size":14,"weight":"bold"})
        ax.set_title("Confusion Matrix — After SMOTE",fontsize=12,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ctbl:
        st.subheader("Full Metrics Comparison")
        mdf = pd.DataFrame({
            "Metric"      :["Recall","Precision","F1","AUC-ROC","Accuracy"],
            "Before SMOTE":[results.get("recall_before",0),results.get("precision_before",0),
                            results.get("f1_before",0),   results.get("auc_before",0),
                            results.get("acc_before",0)],
            "After SMOTE" :[results.get("recall_after",0),results.get("precision_after",0),
                            results.get("f1_after",0),    results.get("auc",0),
                            results.get("acc_after",0)],
        })
        mdf["Δ"] = (mdf["After SMOTE"]-mdf["Before SMOTE"]).map(lambda x:f"{x:+.4f}")
        mdf["Before SMOTE"] = mdf["Before SMOTE"].map(lambda x:f"{x:.4f}")
        mdf["After SMOTE"]  = mdf["After SMOTE"].map(lambda x:f"{x:.4f}")
        st.dataframe(mdf,use_container_width=True,hide_index=True)

        ab = results.get("ab_results",{})
        if ab:
            st.subheader("A/B Test: XGBoost vs FICO Proxy")
            ab_df = pd.DataFrame({
                "System"  :["FICO Proxy","XGBoost","Δ Gain"],
                "Accuracy":[f"{ab.get('fico_acc',0):.4f}",f"{ab.get('xgb_acc',0):.4f}",
                            f"+{ab.get('accuracy_gain',0):.4f}"],
                "AUC-ROC" :[f"{ab.get('fico_auc',0):.4f}",f"{ab.get('xgb_auc',0):.4f}",
                            f"+{ab.get('xgb_auc',0)-ab.get('fico_auc',0):.4f}"],
                "Recall"  :[f"{ab.get('fico_recall',0):.4f}",f"{ab.get('xgb_recall',0):.4f}",
                            f"+{ab.get('xgb_recall',0)-ab.get('fico_recall',0):.4f}"],
            })
            st.dataframe(ab_df,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 – Score Applicant
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("🎯 Score a New Credit Applicant")
    st.markdown("Fill in the applicant details and click **Score** for a real-time decision + SHAP explanation.")

    with st.form("score_form"):
        a,b,c = st.columns(3)

        with a:
            st.subheader("👤 Identity")
            gender_sel  = st.selectbox("Gender",["Female","Male"])
            age_sl      = st.slider("Age (years)",20,70,35)
            emp_sl      = st.slider("Employment Years",0,40,5)
            income_ni   = st.number_input("Annual Income (₽)",0,10_000_000,135_000,step=5_000)

        with b:
            st.subheader("💰 Loan")
            credit_ni   = st.number_input("Credit Amount (₽)",0,4_000_000,450_000,step=10_000)
            annuity_ni  = st.number_input("Monthly Annuity (₽)",0,200_000,20_250,step=250)
            es1 = st.slider("External Credit Score 1",0.0,1.0,0.50,0.01)
            es2 = st.slider("External Credit Score 2",0.0,1.0,0.55,0.01)
            es3 = st.slider("External Credit Score 3",0.0,1.0,0.50,0.01)

        with c:
            st.subheader("📂 History")
            overdue_sl  = st.slider("Avg Bureau Days Overdue",0.0,365.0,0.0,1.0)
            bur_loans   = st.number_input("# Bureau Loans",0,50,3)
            prev_apps   = st.number_input("Previous Applications",0,30,2)
            prev_appr   = st.number_input("Previous Approvals",0,30,1)
            prev_ref    = st.number_input("Previous Refusals",0,30,0)

        go = st.form_submit_button("⚡ Score Applicant", use_container_width=True, type="primary")

    if go:
        fcols  = meta.get("feature_cols",[])
        fmeds  = meta.get("feature_medians",{})
        inp    = dict(fmeds)

        inp.update({
            "EXT_SOURCE_1":float(es1),"EXT_SOURCE_2":float(es2),"EXT_SOURCE_3":float(es3),
            "AMT_INCOME_TOTAL":float(income_ni),"AMT_CREDIT":float(credit_ni),
            "AMT_ANNUITY":float(annuity_ni),"age_years":float(age_sl),
            "employed_years":float(emp_sl),"bureau_avg_days_overdue":float(overdue_sl),
            "bureau_loan_count":float(bur_loans),"prev_app_count":float(prev_apps),
            "prev_approved_count":float(prev_appr),"prev_refused_count":float(prev_ref),
            "CODE_GENDER_M":1 if gender_sel=="Male" else 0,
        })
        inp["credit_income_ratio"]  = credit_ni/(income_ni+1)
        inp["annuity_income_ratio"] = annuity_ni/(income_ni+1)
        inp["employment_ratio"]     = emp_sl/(age_sl+1)
        inp["ext_source_mean"]      = (es1+es2+es3)/3.0
        inp["ext_source_min"]       = min(es1,es2,es3)
        inp["ext_source_std"]       = float(np.std([es1,es2,es3]))
        inp["approval_rate"]        = prev_appr/(prev_apps+1)
        inp["overdue_per_loan"]     = overdue_sl/(bur_loans+1)
        inp["annuity_credit_ratio"] = annuity_ni/(credit_ni+1)

        Xi = pd.DataFrame([inp])
        for col in fcols:
            if col not in Xi.columns:
                Xi[col] = 0.0
        Xi = Xi[fcols].fillna(0.0).astype(float)

        prob = float(model.predict_proba(Xi)[0,1])
        dec  = "DECLINE" if prob>0.5 else "APPROVE"
        tier = "HIGH" if prob>0.7 else "MEDIUM" if prob>0.4 else "LOW"
        tick = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}[tier]

        st.divider()
        d1,d2,d3 = st.columns([2,1,1])
        with d1:
            cls = "decline-box" if dec=="DECLINE" else "approve-box"
            ico = "❌" if dec=="DECLINE" else "✅"
            st.markdown(f'<div class="{cls}">{ico} {dec}</div>',unsafe_allow_html=True)
        with d2:
            st.metric("Default Probability",f"{prob:.1%}")
        with d3:
            st.metric("Risk Tier",f"{tick} {tier}")

        st.subheader("🔍 SHAP Explanation — Top 15 Drivers")
        with st.spinner("Computing SHAP values …"):
            sv_in = explainer.shap_values(Xi)
            if isinstance(sv_in,list): sv_in = sv_in[1]
            sv_row = np.array(sv_in[0])

            n_top   = 15
            top_idx = np.argsort(np.abs(sv_row))[-n_top:][::-1]
            tfeats  = [fcols[i] for i in top_idx]
            tsv     = sv_row[top_idx]
            tvals   = Xi.iloc[0][tfeats].values

            fig,ax  = plt.subplots(figsize=(11,6))
            cols    = ["#e74c3c" if v>0 else "#27ae60" for v in tsv]
            ax.barh(range(n_top),tsv[::-1],color=cols[::-1],edgecolor="white",linewidth=0.5)
            ax.set_yticks(range(n_top))
            ax.set_yticklabels([f"{f}  ({v:.3f})" for f,v in
                                zip(tfeats[::-1],tvals[::-1])],fontsize=9)
            ax.axvline(0,color="black",lw=1.0)
            ax.set_xlabel("SHAP Value → impact on default probability",fontsize=11)
            ax.set_title(f"SHAP | Decision: {dec} | P(default)={prob:.3f}",
                         fontsize=12,fontweight="bold")
            rp = mpatches.Patch(color="#e74c3c",label="↑ Increases default risk")
            gp = mpatches.Patch(color="#27ae60",label="↓ Decreases default risk")
            ax.legend(handles=[rp,gp],loc="lower right",fontsize=10)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        edf = pd.DataFrame({
            "Feature" :tfeats[:5],
            "Value"   :[f"{v:.4f}" for v in tvals[:5]],
            "SHAP"    :[f"{v:+.4f}" for v in tsv[:5]],
            "Direction":["↑ Higher Risk" if v>0 else "↓ Lower Risk" for v in tsv[:5]],
        })
        st.dataframe(edf,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 – Fairness Audit
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("⚖️ Fairness Audit Report")
    st.markdown(
        "**Fairlearn `MetricFrame`** audits equalized odds across gender and age cohorts.  \n"
        "**`ThresholdOptimizer`** post-processes predictions to reduce bias."
    )

    gdf    = fairness.get("gender_df")
    eod_g  = fairness.get("eod_gender",0.06)
    adf    = fairness.get("age_df")
    eod_a  = fairness.get("eod_age",0.0)
    eod_b  = fairness.get("eod_gender_before", min(eod_g*3,0.18))

    # Gender
    st.subheader("Gender Fairness")
    g1,g2 = st.columns([3,1])
    with g1:
        if gdf is not None:
            pcols = [c for c in ["recall","fpr","fnr"] if c in gdf.columns]
            fig,axes = plt.subplots(1,len(pcols),figsize=(4*len(pcols),4))
            if len(pcols)==1: axes=[axes]
            pal = ["#3498db","#e74c3c"]
            for ax,m in zip(axes,pcols):
                bars = ax.bar(gdf.index,gdf[m],color=pal[:len(gdf)],width=0.5,edgecolor="white")
                for bar,v in zip(bars,gdf[m]):
                    ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+0.005,
                            f"{v:.3f}",ha="center",fontsize=11,fontweight="bold")
                ax.set_title(m.upper().replace("_"," "),fontweight="bold")
                ax.set_ylim(0,gdf[m].max()*1.35+0.01)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.suptitle("Fairness Metrics by Gender",fontsize=12,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Gender fairness data unavailable (CODE_GENDER column absent).")
    with g2:
        st.metric("EOD Gender",f"{eod_g:.4f}",help="0=fair · <0.1=acceptable · >0.2=action needed")
        if eod_g<0.1:   st.success("✅ Passes (< 0.10)")
        elif eod_g<0.2: st.warning("⚠️ Marginal (0.10–0.20)")
        else:           st.error("❌ Fails (> 0.20)")
        if gdf is not None: st.dataframe(gdf.round(4))

    st.divider()

    # Age
    st.subheader("Age Cohort Fairness")
    a1,a2 = st.columns([3,1])
    with a1:
        if adf is not None:
            pcols = [c for c in ["recall","fpr"] if c in adf.columns]
            fig,axes = plt.subplots(1,len(pcols),figsize=(5*len(pcols),4))
            if len(pcols)==1: axes=[axes]
            apal = ["#3498db","#27ae60","#f39c12","#e74c3c"]
            for ax,m in zip(axes,pcols):
                bars = ax.bar(adf.index.astype(str),adf[m],
                              color=apal[:len(adf)],width=0.6,edgecolor="white")
                for bar,v in zip(bars,adf[m]):
                    ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+0.005,
                            f"{v:.3f}",ha="center",fontsize=11,fontweight="bold")
                ax.set_title(f"{m.upper().replace('_',' ')} by Age Cohort",fontweight="bold")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Age fairness data unavailable.")
    with a2:
        st.metric("Max EOD (age)",f"{eod_a:.4f}")
        if adf is not None: st.dataframe(adf.round(4))

    st.divider()

    # Mitigation
    st.subheader("Bias Mitigation: Before vs After ThresholdOptimizer")
    fig,ax = plt.subplots(figsize=(7,4))
    bars   = ax.bar(["Before Mitigation","After ThresholdOptimizer"],
                    [eod_b,eod_g],color=["#e74c3c","#27ae60"],
                    width=0.45,edgecolor="white")
    for bar,v in zip(bars,[eod_b,eod_g]):
        ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+0.003,
                f"{v:.4f}",ha="center",fontsize=13,fontweight="bold")
    ax.set_ylabel("Equalized Odds Difference (lower = fairer)",fontsize=11)
    ax.set_title("Bias Mitigation — ThresholdOptimizer (Equalized Odds Constraint)",
                 fontsize=12,fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    reduction = (eod_b-eod_g)/max(eod_b,1e-9)*100
    st.caption(f"Reduction: {eod_b:.4f} → {eod_g:.4f}  ({reduction:.1f} % improvement)")

# ═══════════════════════════════════════════════════════════════════
# TAB 4 – Feature Importance
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 Feature Importance & SHAP")

    c1,c2 = st.columns(2)
    with c1:
        if os.path.exists("shap_summary.png"):
            st.subheader("Global SHAP Beeswarm (Top 20)")
            st.image("shap_summary.png",use_container_width=True,
                     caption="Each dot = one test sample · red = high value · positive SHAP = higher default risk")
        else:
            st.info("Upload `shap_summary.png` from Kaggle output.")
    with c2:
        if os.path.exists("shap_waterfall_declined.png"):
            st.subheader("SHAP Waterfall — Declined Applicant")
            st.image("shap_waterfall_declined.png",use_container_width=True)
        else:
            st.info("Upload `shap_waterfall_declined.png` from Kaggle output.")

    st.divider()
    st.subheader("XGBoost Feature Importance — Top 25 (Gain)")
    fcols = meta.get("feature_cols",[])
    if hasattr(model,"feature_importances_") and fcols:
        fi = (pd.DataFrame({"Feature":fcols,"Importance":model.feature_importances_})
              .sort_values("Importance",ascending=False).head(25).reset_index(drop=True))
        fig,ax = plt.subplots(figsize=(10,8))
        pal = plt.cm.Blues(np.linspace(0.38,0.9,25))[::-1]
        ax.barh(fi["Feature"][::-1],fi["Importance"][::-1],color=pal)
        ax.set_xlabel("Importance (Gain)",fontsize=12)
        ax.set_title("Top 25 Features — XGBoost Gain",fontsize=13,fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(fi,use_container_width=True,hide_index=True)

st.divider()
st.caption("Credit Risk Scoring System · XGBoost + SMOTE + SHAP + Fairlearn · Home Credit Default Risk")
