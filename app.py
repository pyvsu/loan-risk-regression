import re

import streamlit as st

from predict import META, explain, percentile_rank, predict, risk_band

                                   
def label(name: str) -> str:
    """'CreditCardUtilizationRate' -> 'Credit Card Utilization Rate'"""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


st.set_page_config(page_title="Loan Risk Score", page_icon="📊")
st.title("Loan Risk Score Predictor")
st.caption(
    "Linear regression (v3, 25 features) trained on a synthetic Kaggle dataset. "
    "A learning project, not for real credit decisions."
)
st.write("Every field starts at the typical borrower. Change what you know, then press the button.")

applicant = {}
with st.form("applicant_form"):
    left, right = st.columns(2)

    for i, (name, m) in enumerate(META["numeric"].items()):
        with (left if i % 2 == 0 else right):
            if m["is_int"] and m["min"] == 0 and m["max"] == 1:          # a yes/no flag
                applicant[name] = int(st.checkbox(label(name)))
            elif m["is_int"]:
                applicant[name] = st.number_input(
                    label(name), int(m["min"]), int(m["max"]), int(m["median"]), step=1
                )
            else:
                applicant[name] = st.number_input(
                    label(name), float(m["min"]), float(m["max"]), float(m["median"])
                )

    for i, (name, info) in enumerate(META["categorical"].items()):
        with (left if i % 2 == 0 else right):
            options = info["options"]
            applicant[name] = st.selectbox(label(name), options, index=options.index(info["default"]))

    submitted = st.form_submit_button("Predict risk score")

if submitted:
    score = predict(applicant)
    target = META["target"]

    st.metric("Predicted risk score (higher = riskier)", f"{score:.1f}")
    band = risk_band(score)
    cuts = target["percentiles"]
    show = {"Low": st.success, "Medium": st.warning, "High": st.error}[band]
    show(f"{band} risk compared with the training borrowers")
    st.caption(
        f"Low: below {cuts[33]:.1f} | Medium: {cuts[33]:.1f} to {cuts[67]:.1f} | High: above {cuts[67]:.1f}. "
        "These cutoffs split the training borrowers into thirds. They are not an industry standard."
    )
    st.write(
        f"Riskier than about {percentile_rank(score):.0f}% of borrowers in the training data "
        f"(scores ran from {target['min']:.1f} to {target['max']:.1f})."
    )
    st.write(
        f"On borrowers the model had not seen, it missed by about "
        f"{META['metrics_test']['mae']:.1f} points on average."
    )

    st.subheader("What moved the score")
    st.caption("Points added (+) or removed (-) compared with an average borrower.")
    st.bar_chart(explain(applicant))