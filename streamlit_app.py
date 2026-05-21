import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random

st.set_page_config(
    page_title="SM2 Logistische Regressie Trainer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- STYLING ----------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 0.5rem;
}
h1 {
    margin-bottom: 0rem;
}
h2, h3 {
    margin-top: 0.3rem;
}
.small-box {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 0.7rem;
    background-color: #fafafa;
}
.feedback-good {
    border: 1px solid #9ee6a3;
    background-color: #eaffea;
    padding: 0.8rem;
    border-radius: 10px;
}
.feedback-bad {
    border: 1px solid #ffb3b3;
    background-color: #fff0f0;
    padding: 0.8rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------- DATA ----------
def maak_data():
    n = random.randint(180, 320)

    uren = np.random.normal(8, 3, n).clip(0, 20)
    voorkennis = np.random.normal(6, 1.5, n).clip(0, 10)
    angst = np.random.normal(5, 2, n).clip(0, 10)

    b0 = random.uniform(-6, -3)
    b_uren = random.uniform(0.15, 0.45)
    b_voorkennis = random.uniform(0.15, 0.65)
    b_angst = random.uniform(-0.50, -0.10)

    logit_p = b0 + b_uren * uren + b_voorkennis * voorkennis + b_angst * angst
    p = 1 / (1 + np.exp(-logit_p))
    geslaagd = np.random.binomial(1, p, n)

    return pd.DataFrame({
        "geslaagd": geslaagd,
        "uren_studie": uren,
        "voorkennis": voorkennis,
        "angst": angst
    })


def fit_model(df):
    X = sm.add_constant(df[["uren_studie", "voorkennis", "angst"]])
    y = df["geslaagd"]
    return sm.Logit(y, X).fit(disp=False)


def format_p(p):
    return "< .001" if p < .001 else f"{p:.3f}"


def output_tabel(model):
    params = model.params
    se = model.bse
    wald = (params / se) ** 2
    pvalues = model.pvalues
    odds = np.exp(params)
    ci = np.exp(model.conf_int())

    tabel = pd.DataFrame({
        "B": params,
        "SE": se,
        "Wald": wald,
        "p": pvalues,
        "Exp(B)": odds,
        "BI laag": ci[0],
        "BI hoog": ci[1]
    })

    tabel["p"] = tabel["p"].apply(format_p)
    return tabel.round(3)


# ---------- VRAGEN ----------
def genereer_vraag(tabel_raw, model):
    predictor = random.choice(["uren_studie", "voorkennis", "angst"])
    rij = tabel_raw.loc[predictor]

    vraagtype = random.choice([
        "expb",
        "pwaarde",
        "b_teken",
        "wald",
        "logit",
        "linear_probability",
        "likelihood_ratio"
    ])

    if vraagtype == "expb":
        return {
            "vraag": f"Exp(B) voor **{predictor}** is **{np.exp(rij['B']):.2f}**. Wat betekent dit?",
            "opties": [
                "De kans verandert met precies dit percentage.",
                "De odds worden per 1 punt stijging met deze factor vermenigvuldigd.",
                "De predictor is automatisch niet significant.",
                "De afhankelijke variabele is continu."
            ],
            "goed": 1,
            "uitleg": "Exp(B) is de odds ratio. Die zegt hoe de odds veranderen per 1 punt stijging in de predictor."
        }

    if vraagtype == "pwaarde":
        significant = rij["p"] < .05
        return {
            "vraag": f"De p-waarde voor **{predictor}** is **{format_p(rij['p'])}**. Wat concludeer je bij α = .05?",
            "opties": [
                "De predictor draagt significant bij aan het model.",
                "De predictor draagt niet significant bij aan het model.",
                "De odds ratio is precies 1.",
                "De uitkomstvariabele is niet binary."
            ],
            "goed": 0 if significant else 1,
            "uitleg": "Vergelijk p met α = .05. Is p kleiner dan .05, dan verwerp je H0: B = 0."
        }

    if vraagtype == "b_teken":
        positief = rij["B"] > 0
        return {
            "vraag": f"De B-coëfficiënt voor **{predictor}** is **{rij['B']:.2f}**. Wat zegt het teken?",
            "opties": [
                "Bij een positieve B nemen de log odds toe.",
                "Bij een negatieve B nemen de log odds af.",
                "Het teken zegt niets over richting.",
                "B is hetzelfde als de kans."
            ],
            "goed": 0 if positief else 1,
            "uitleg": "B werkt op de logit/log odds. Positief betekent hogere log odds; negatief betekent lagere log odds."
        }

    if vraagtype == "wald":
        return {
            "vraag": "Wat toetst de **Wald-statistic** in deze regressietabel?",
            "opties": [
                "Of B = 0 voor een predictor.",
                "Of de ruwe scores normaal verdeeld zijn.",
                "Of de kans altijd groter is dan .50.",
                "Of alle observaties identiek zijn."
            ],
            "goed": 0,
            "uitleg": "De Wald-statistic toetst per predictor of de regressiecoëfficiënt afwijkt van nul."
        }

    if vraagtype == "logit":
        return {
            "vraag": "Wat is de **logit** in logistische regressie?",
            "opties": [
                "De gewone kans p.",
                "De odds zonder logaritme.",
                "De logaritme van de odds: log(p / (1-p)).",
                "De gemiddelde waarde van Y."
            ],
            "goed": 2,
            "uitleg": "De logit is log(p / (1-p)). Daardoor kan een kansmodel lineair worden geschreven."
        }

    if vraagtype == "linear_probability":
        return {
            "vraag": "Waarom gebruiken we liever geen **linear probability model** voor een binary uitkomst?",
            "opties": [
                "Omdat voorspelde kansen onder 0 of boven 1 kunnen uitkomen.",
                "Omdat binary variabelen nooit getallen zijn.",
                "Omdat logistische regressie geen predictors gebruikt.",
                "Omdat lineaire regressie altijd zonder intercept werkt."
            ],
            "goed": 0,
            "uitleg": "Een gewone regressielijn is niet begrensd tussen 0 en 1. Logistische regressie lost dat op."
        }

    return {
        "vraag": f"De likelihood-ratio p-waarde van het model is **{format_p(model.llr_pvalue)}**. Wat toetst deze test?",
        "opties": [
            "Of het model met predictors beter past dan het nulmodel.",
            "Of alle individuele studenten hetzelfde scoorden.",
            "Of Exp(B) altijd kleiner is dan 1.",
            "Of de afhankelijke variabele normaal verdeeld is."
        ],
        "goed": 0,
        "uitleg": "De likelihood-ratio test vergelijkt een model met predictors met een eenvoudiger nulmodel."
    }


def ruwe_tabel(model):
    params = model.params
    se = model.bse
    wald = (params / se) ** 2
    pvalues = model.pvalues
    odds = np.exp(params)
    ci = np.exp(model.conf_int())

    return pd.DataFrame({
        "B": params,
        "SE": se,
        "Wald": wald,
        "p": pvalues,
        "Exp(B)": odds,
        "BI laag": ci[0],
        "BI hoog": ci[1]
    })


# ---------- SESSION ----------
def nieuwe_dataset():
    df = maak_data()
    model = fit_model(df)
    tabel_raw = ruwe_tabel(model)

    st.session_state.df = df
    st.session_state.model = model
    st.session_state.tabel_raw = tabel_raw
    st.session_state.tabel = output_tabel(model)
    st.session_state.vraag = genereer_vraag(tabel_raw, model)
    st.session_state.beantwoord = False
    st.session_state.laat_feedback = None


def nieuwe_vraag():
    st.session_state.vraag = genereer_vraag(st.session_state.tabel_raw, st.session_state.model)
    st.session_state.beantwoord = False
    st.session_state.laat_feedback = None


if "df" not in st.session_state:
    st.session_state.score = 0
    st.session_state.aantal = 0
    nieuwe_dataset()


# ---------- PLOT ----------
def plot_logistische_curve(df, model):
    params = model.params
    x_range = np.linspace(0, 20, 200)

    gemiddelde_voorkennis = df["voorkennis"].mean()
    gemiddelde_angst = df["angst"].mean()

    logit = (
        params["const"]
        + params["uren_studie"] * x_range
        + params["voorkennis"] * gemiddelde_voorkennis
        + params["angst"] * gemiddelde_angst
    )

    kans = 1 / (1 + np.exp(-logit))

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    jitter = np.random.normal(0, 0.025, len(df))

    ax.scatter(df["uren_studie"], df["geslaagd"] + jitter, alpha=0.25, s=15)
    ax.plot(x_range, kans, linewidth=2.5)

    ax.set_title("Kans op slagen bij uren studie", fontsize=11)
    ax.set_xlabel("Uren studie per week", fontsize=9)
    ax.set_ylabel("Kans", fontsize=9)
    ax.set_ylim(-0.08, 1.08)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()

    return fig


# ---------- LAYOUT ----------
st.title("SM2 – Logistische regressie trainer")
st.caption("Hoofdstuk 15: binary uitkomst, logit, odds ratio, Wald en likelihood-ratio test")

col_left, col_right = st.columns([0.95, 1.55], gap="large")

with col_left:
    st.subheader("Quizvraag")

    vraag = st.session_state.vraag
    st.markdown(vraag["vraag"])

    keuze = st.radio(
        "Kies je antwoord:",
        vraag["opties"],
        index=None,
        disabled=st.session_state.beantwoord
    )

    if st.button("Controleer antwoord", disabled=keuze is None or st.session_state.beantwoord):
        st.session_state.aantal += 1
        gekozen_index = vraag["opties"].index(keuze)

        if gekozen_index == vraag["goed"]:
            st.session_state.score += 1
            st.session_state.laat_feedback = "goed"
        else:
            st.session_state.laat_feedback = "fout"

        st.session_state.beantwoord = True
        st.rerun()

    if st.session_state.laat_feedback == "goed":
        st.markdown(
            f"<div class='feedback-good'><b>Goed!</b><br>{vraag['uitleg']}</div>",
            unsafe_allow_html=True
        )

    if st.session_state.laat_feedback == "fout":
        goed_antwoord = vraag["opties"][vraag["goed"]]
        st.markdown(
            f"<div class='feedback-bad'><b>Niet helemaal.</b><br>"
            f"Goed antwoord: {goed_antwoord}<br><br>{vraag['uitleg']}</div>",
            unsafe_allow_html=True
        )

    st.markdown(f"### Score: {st.session_state.score}/{st.session_state.aantal}")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Nieuwe vraag", use_container_width=True):
            nieuwe_vraag()
            st.rerun()

    with b2:
        if st.button("Nieuwe dataset", use_container_width=True):
            nieuwe_dataset()
            st.rerun()

with col_right:
    top1, top2 = st.columns([1.35, 0.8], gap="medium")

    with top1:
        st.subheader("Regressietabel")
        st.dataframe(
            st.session_state.tabel,
            width="stretch",
            height=205
        )

    with top2:
        st.subheader("Model")
        model = st.session_state.model
        lr_chi2 = 2 * (model.llf - model.llnull)

        st.metric("LR χ²", f"{lr_chi2:.2f}")
        st.metric("LLR p", format_p(model.llr_pvalue))
        st.metric("Pseudo R²", f"{model.prsquared:.3f}")

    bottom1, bottom2 = st.columns([1.2, 0.8], gap="medium")

    with bottom1:
        st.subheader("Grafiek")
        fig = plot_logistische_curve(st.session_state.df, st.session_state.model)
        st.pyplot(fig, use_container_width=True)

    with bottom2:
        st.subheader("Kernbegrippen")
        st.markdown("""
**Binary**: uitkomst met twee categorieën.  
**Logit**: log(p / (1-p)).  
**Exp(B)**: odds ratio.  
**Wald**: toetst of B = 0.  
**LR-test**: vergelijkt model met nulmodel.  
**Linear probability model**: kan kansen < 0 of > 1 voorspellen.
""")
