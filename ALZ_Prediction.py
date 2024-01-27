import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lime.lime_tabular import LimeTabularExplainer
import shap
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

def about_proj():
    st.header("Project Title")

    st.markdown(
        """A Customized Machine Learning and Deep Learning Model for Predicting the Risk of Alzheimer’s Disease at an 
        Early Stage Using Magnetic Resonance Imaging Data.""")
   
    st.header("Funding Details")


    st.markdown(
        """The research leading to these results has received funding from 
        the Research Council (TRC) of the Sultanate of Oman under the Block Funding Program BFP/RGP/ICT/21/148""")
    st.header("Project Team")

    st.markdown("Dr. Abraham Varghese")
    st.markdown("Dr. Vinu Sherimon")
    st.markdown("Dr. Ben George")
    st.markdown("Dr. Prashanth Gouda (MBBS, MD, MRCPCH) *")
    st.markdown("\n")
    st.markdown("**Institutional Affiliation:**")
    st.markdown("University of Technology and Applied Sciences (UTAS), Muscat and")
    st.markdown("*National University of Science and Technology (NUST), Oman.")
    
    st.header("Source of Data for the Models")

    st.markdown(
        """The data utilized in this study was obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI)
        through their standardized datasets. ADNI is a major research initiative focused on understanding and identifying 
        biomarkers for Alzheimer's disease. Access to the data can be requested through the 
        ADNI website (https://ida.loni.usc.edu/login.jsp?project=ADNI) after registering and obtaining approval.""")

def about_ALZ():
    
    st.header("Understanding Alzheimer's Disease")

    st.markdown(
        """
        Alzheimer's disease, a progressive neurodegenerative disorder that gradually erodes memory, cognitive functions, and daily activities, impacts millions of individuals worldwide. Early detection is crucial for timely intervention and personalized care.
        """
    )

    st.header("AI in Prediction of Alzheimer's Disease")

    st.markdown(
        """
        The Deciphering of Alzheimer's Disease Prediction with Explanations application harnesses the power of Artificial Intelligence to predict Alzheimer's disease with precision. Using a Random Forest classifier, the app considers vital input features such as 'AGE,' 'CDRSB,' 'ADAS11,' 'ADAS13,' 'MMSE,' 'RAVLT_immediate,' 'RAVLT_learning,' 'FAQ,' and 'Hippocampus,'  to classify individuals into one of three categories: 'DEMENTIA,' 'MCI' (Mild Cognitive Impairment), or 'Normal.'
        """
    )


    st.header("Transparent Explanations with LIME")

    st.markdown(
        """
        Understanding the rationale behind every prediction is paramount. Our application employs LIME (Local Interpretable Model-agnostic Explanations) to provide clear, local insights into why the model made a specific prediction for an individual. LIME generates easily understandable explanations, allowing users to grasp the influential features and factors contributing to the predicted class.
        """
    )

    st.header("SHAP Values for Comprehensive Understanding")

    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) values enhance interpretability by quantifying the impact of each feature on the model's output. Our application utilizes SHAP to create interactive visualizations, including Summary Plots, force plots, and decision plots. This comprehensive approach empowers users to explore the relative importance of different features in predicting each class, fostering a deeper understanding of the model's decision logic.
        """
    )
    

    st.header("Early Detection and Progression Prediction")
    st.subheader("Predict Current Status")
    st.markdown(
        """
       This AI-powered system allows you to predict the current cognitive status of an individual. By inputting relevant information such as age and current Clinical Dementia Rating Scale Sum of Boxes (CDRSB) value, the model provides insights into whether the person is currently in a normal cognitive state, is experiencing Mild Cognitive Impairment (MCI), or has progressed to Dementia. This early detection capability enables timely interventions and personalized care plans.
        """
    )
    st.subheader("Predict Progression")
    st.markdown(
        """
       Understanding the progression of cognitive health is crucial for effective healthcare planning. This system also offers the capability to predict the progression of an individual based on their last checkup. This includes predicting whether the person will remain in a normal cognitive state, MCI, Dementia, transition from MCI to Dementia, or experience other changes such as moving from Normal (NL) to MCI. 
        """
    )


    st.header("User-Friendly Input and Insights")
    st.markdown(
        """
        Inputting and interpreting data is simplified through our user-friendly interface. Users can effortlessly input relevant features, obtain predictions for Alzheimer's disease classes, and delve into detailed explanations.
        """
    )

    # Call to Action
    #st.button("Experience AI-powered Alzheimer's Disease Prediction")
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 36px">Experience the power of  interpretable AI in Alzheimer's disease prediction with 'Deciphering Alzheimer's Disease Prediction with Explanations.' </h2>
        """,
        unsafe_allow_html=True
    )

def about_features():
    st.title("Know about the Features used in this Research")

    features_info = {
        "CDRSB (Clinical Dementia Rating Scale Sum of Boxes)": (
            "CDRSB is a clinical tool used to assess the severity of dementia. "
            "It combines ratings from multiple domains, including memory, orientation, judgment, and problem-solving. "
            "The sum of scores provides an overall measure of cognitive and functional impairment."
        ),
        "ADAS11 (Alzheimer's Disease Assessment Scale - 11 items)": (
            "ADAS11 is a cognitive assessment tool used to measure cognitive dysfunction in individuals with Alzheimer's disease. "
            "It includes 11 items that assess various cognitive functions, such as memory and language."
        ),
        "ADAS13 (Alzheimer's Disease Assessment Scale - 13 items)": (
            "ADAS13 is an expanded version of ADAS11, including additional items to assess cognitive function more comprehensively. "
            "It provides a broader evaluation of cognitive abilities affected by Alzheimer's disease."
        ),
        "MMSE (Mini-Mental State Examination)": (
            "MMSE is a widely used screening tool for cognitive impairment. "
            "It assesses various cognitive domains, including orientation, memory, attention, and language. "
            "Scores on the MMSE can indicate the severity of cognitive impairment."
        ),
        "RAVLT_immediate (Rey Auditory Verbal Learning Test - Immediate Recall)": (
            "RAVLT_immediate assesses verbal learning and immediate recall. "
            "It involves presenting a list of words to the individual, who is then asked to recall the words immediately after."
        ),
        "RAVLT_learning (Rey Auditory Verbal Learning Test - Learning Score)": (
            "RAVLT_learning measures the ability to learn and remember verbal information over several trials."
        ),
        "FAQ (Functional Activities Questionnaire)": (
            "FAQ assesses an individual's ability to perform daily activities and tasks. "
            "It helps evaluate functional impairment and the impact of cognitive decline on daily life."
        ),
        "Hippocampus Volume": (
            "The hippocampus is a region in the brain associated with memory and learning. "
            "Changes in hippocampal volume are often observed in conditions affecting memory."
        ),
        "Feature used in Predicting the Progression": (
            "Feature Names end with cur: The values recorded during the current visit.                       "
            "Feature Names end with diff: (values recorded during the current visit - values recorded during the previous visit) / number of months"
        )
    }

    for feature, description in features_info.items():
        st.subheader(feature)
        st.write(description)
        st.markdown("---") 

def home():

    st.title("Demystifying Alzheimer's Disease Prediction and Progression")

    image = Image.open("alz_bg.png")

    #resized_image = image.resize((800, image.size[1]))
    st.image(image)
    #st.image(resized_image, width=800)

    st.markdown(
        """
        <p style="text-align: center; font-size: 24px">Harnessing the power of Artificial Intelligence for early detection and transparent explanations.</p>
        """,
        unsafe_allow_html=True
    )

    if st.button("Project Details - Members - Funding"):
        about_proj()   
        
    if st.button("Explore About the Application"):
        about_ALZ()

    
    if st.button("Discover About the Features"):
        about_features()
        
def predict_explain_current():
    filename = 'rfc_model.sav'

    # Initialize session state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False


    #st.markdown(hide_st_style, unsafe_allow_html=True)
    st.title("Deciphering  of Alzheimer's Disease Prediction with Explanations - Current")


    @st.cache_data
    def load_data():
        alz_df = pd.read_csv("alz_balanc.csv")
        loaded_model1 = pickle.load(open(filename, 'rb'))
        X1 = alz_df.iloc[:,:-1]
        y1 = alz_df.iloc[:,-1]
        return loaded_model1, X1, y1
    loaded_model, X, y = load_data()

    age = st.number_input("Age of the person",50,100,step=1,value=st.session_state.age, help = "Current age of the individual")
    cdr = st.number_input("CDRSB(0-20)",0.0,20.0,step=0.1,value=st.session_state.cdr, help="Clinical Dementia Rating Scale Sum of Boxes")
    ads11 = st.number_input('ADAS11 (0 - 75)',0.0,75.0,step=0.1,value=st.session_state.ads11,help="Alzheimer's Disease Assessment Scale (11 items) ")
    ads13 = st.number_input("ADAS13 (0 - 85)",0.0,85.0,step=0.1,value=st.session_state.ads13,help="Alzheimer's Disease Assessment Scale (13 items) ")
    mmse = st.number_input("MMSE (0 - 30)",0.0,30.0,step=0.1,value=st.session_state.mmse, help="Mini-Mental State Examination Score")
    rai = st.number_input("RAVLT_immediate (0 - 75)",0.0,75.0,step=0.1,value=st.session_state.rai, help = "Rey Auditory Verbal Learning Test - Immediate Recall")
    ral = st.number_input("RAVLT_learning (-5 - 15)",-5.0,15.0,step=0.1,value=st.session_state.ral, help="Rey Auditory Verbal Learning Test - Learning Score")
    faq = st.number_input("FAQ (0 - 30)",0.0,30.0,step=0.1,value=st.session_state.faq,help="Functional Activities Questionnaire Score")
    hp = st.number_input("Hippocampus (2200,11300)",2200,11300,step=1,value=st.session_state.hp,help = "Hippocampal volume ")
   
    st.session_state.age = age
    st.session_state.cdr = cdr
    st.session_state.ads11 = ads11
    st.session_state.ads13 = ads13
    st.session_state.mmse = mmse
    st.session_state.rai = rai
    st.session_state.ral = ral
    st.session_state.faq = faq
    st.session_state.hp = hp

    dic = {'AGE':age, 'CDRSB':cdr, 'ADAS11':ads11, 'ADAS13':ads13, 'MMSE':mmse,
           'RAVLT_immediate':rai, 'RAVLT_learning':ral, 'FAQ':faq, 'Hippocampus':hp}
    df = pd.DataFrame([dic])
    res = loaded_model.predict(df)

    class_nam=['DEMENTIA', 'MCI', 'NL']

    if st.button('Predict - Current'):
        st.session_state.button_clicked = not st.session_state.button_clicked


    if st.session_state.button_clicked:
        if res[0] == 'DEMENTIA':
            st.subheader('Predicted Class - DEMENTIA')
        elif res[0] == 'NL':
            st.subheader('Predicted Class - NL- Normal')
        else:
            st.subheader('Predicted Class - MCI - Mild Cognitive Impairment')

        st.write(f"Probability of predicting {res[0]} is {max(loaded_model.predict_proba(df)[0])} ")
        sel = st.radio('Choose the Explainability Principle for Current',options=["LIME","SHAP"])
        #st.session_state.sel = sel
        if sel == "LIME":
            explainer = LimeTabularExplainer(X.values, 
                                             feature_names=X.columns, 
                                             class_names=['DEMENTIA', 'MCI', 'NL'], random_state=np.random.seed(10))


            instance_to_explain = df.iloc[0]

            prediction = loaded_model.predict_proba(df)
            predicted_class = class_nam[np.argmax(prediction)]
            explanation = explainer.explain_instance(instance_to_explain.values, 
                                                         loaded_model.predict_proba, 
                                                         num_features=len(X.columns), 
                                                         top_labels=len(explainer.class_names), 
                                                          labels=[predicted_class])
            st.title('Local Interpretable Model-agnostic Explanations (LIME) for Model Prediction',
                     help="LIME is a technique used to explain the predictions of machine learning models locally. It provides insights into how the model's prediction changes for a specific instance by perturbing the input features.")
            #st.subheader(f'Predicted Class: {predicted_class}')

            # Display Lime explanation details

            for label in explanation.available_labels():
                st.subheader(f'LIME Explanation Chart for Class: {class_nam[label]}',
                            help = "chart represents the impact of different features on the model's prediction. Positive weights indicate a feature's contribution towards the predicted class, while negative weights indicate a feature's contribution towards other classes.")
                fig = explanation.as_pyplot_figure(label)
                st.pyplot(fig)

            # Display feature importance
            st.subheader(f'LIME Feature Importance for the predicted class {predicted_class}',
                        help = "Feature importance shows the contribution of each feature to the predicted class.")
            feature_importance = explanation.as_list(label=np.argmax(prediction))
            if len(feature_importance) > 0:
                st.write(pd.DataFrame(feature_importance, columns=['Feature', 'Weight']))
            else:
                st.write("No feature importance data available for the predicted class.")



            # Display perturbation analysis
            st.subheader('LIME Perturbation Analysis')
            feature_mapping = {i: feature_name for i, feature_name in enumerate(X.columns)}
            perturbation_data = explanation.as_map()
            for label, weights in perturbation_data.items():
                st.subheader(f'Perturbation Analysis for Class: {class_nam[label]}',
                            help = """Perturbation analysis displays the impact of different feature values on the model's prediction.

        A positive weight suggests that an increase in the feature value contributes positively to the predicted class.
        Higher values of this feature are associated with a higher likelihood of the predicted class.

        A negative weight indicates that an increase in the feature value contributes negatively to the predicted class.
        Lower values of this feature are associated with a higher likelihood of the predicted class.

        The magnitude of the weight reflects the strength of the feature's impact on the model's prediction.
        A larger magnitude indicates a more significant influence.

        Comparing weights across different features offers insights into the relative importance of each feature for the given instance.
        """)
                #df_perturbation = pd.DataFrame(weights, columns=['Feature', 'Weight'])
                #st.write(df_perturbation)
                df_perturbation = pd.DataFrame(weights, columns=['Feature Number', 'Weight'])
                df_perturbation['Feature Name'] = df_perturbation['Feature Number'].map(feature_mapping)

                # Display the DataFrame with feature names
                st.write(df_perturbation[['Feature Name', 'Weight']])
        else:

            st.title("SHapley Additive exPlanations (SHAP)",help = """SHAP values aim to fairly distribute the contribution of each feature to the model's prediction across all possible combinations of features. """)
                # SHAP Explanation
            shexplainer = shap.Explainer(loaded_model)
            shap_values = shexplainer.shap_values(df)
            np.random.seed(10)


            # Summary Plot
            st.subheader('Summary Plot', help = """The SHAP summary plot provides an overview of feature importance for each class.
            It displays the magnitude and direction of the impact of each feature on the model's output.
            Features with positive SHAP values contribute positively to the predicted class and negative SHAP values contribute 
            negatively to the predicted class.""")
            fig_summary = shap.summary_plot(shap_values, df, show=False,class_names=class_nam)
            st.pyplot(fig_summary)



            # Force Plot
            st.subheader('Force Plot',help = """Force plot shows how each feature contributes to the prediction of the 
            selected class for a given instance. Features with positive contributions increase the likelihood of the 
            predicted class and features with negative contributions decrease the likelihood of the predicted class.
        Red bars represent higher feature values, while blue bars represent lower feature values.""")
            i=0
            for f in class_nam:
                st.write(f"Features contributed for the class {f} ")
                fp = shap.plots.force(shexplainer.expected_value[i], shap_values[i], df, matplotlib = True)
                st.pyplot(fp)
                i+=1

            # Decision Plot
            st.subheader("Decision Plot",help = """Decision plot shows the decision path of the model for a given instance, highlighting the contributions of different features.

        Feature Contributions: Observe how different features influence the decision at various points.

        Positive and Negative Contributions: Positive contributions increase the likelihood of the predicted class, while negative contributions decrease it.""")
            dp = shap.decision_plot(shexplainer.expected_value[1], shap_values[1], df.columns)
            st.pyplot(dp)
    
# Page 3 - Predict - Explain Progression
def predict_explain_progression():
    #st.title("Predict - Explain Progression Page")
    filename = 'rfc_model4_X.sav'

    # Initialize session state
    if 'button_clicked1' not in st.session_state:
        st.session_state.button_clicked1 = False


    #st.markdown(hide_st_style, unsafe_allow_html=True)
    st.title("Deciphering  of Alzheimer's Disease Prediction with Explanations - Progression")


    @st.cache_data
    def load_data1():
        alz_df = pd.read_csv("merged3_diff_oversa.csv")
        loaded_model2 = pickle.load(open(filename, 'rb'))
        X2 = alz_df.iloc[:,:-1]
        y2 = alz_df.iloc[:,-1]
        return loaded_model2, X2, y2
    loaded_model2, X2, y2 = load_data1()

    columns = st.columns(2)
    with columns[0]:
        agec = st.number_input("Current Age of the Person",50,100,step=1,value=st.session_state.agec, help = "Current age of the individual")
    with columns[1]:
        month = st.number_input("Enter the number of Months since last Assessment",1,180,step=1,value=st.session_state.month, help = "Number of months from the previous visit to the current")
    with columns[0]:
        cdrc = st.number_input("Current CDRSB(0-20)",0.0,20.0,step=0.1,value=st.session_state.cdrc, help="Current Clinical Dementia Rating Scale Sum of Boxes")
    with columns[1]:
        cdrp = st.number_input("Previous Visit CDRSB(0-20)",0.0,20.0,step=0.1,value=st.session_state.cdrp, help="Clinical Dementia Rating Scale Sum of Boxes  during previous visit")
    with columns[0]:
        ads11c = st.number_input('Current ADAS11 (0 - 75)',0.0,75.0,step=0.1,value=st.session_state.ads11c,help="Current Alzheimer's Disease Assessment Scale (11 items) ")
    with columns[1]:
        ads11p = st.number_input('Previous Visit ADAS11 (0 - 75)',0.0,75.0,step=0.1,value=st.session_state.ads11p,help="Alzheimer's Disease Assessment Scale (11 items) during previous visit ")
    with columns[0]:
        ads13c = st.number_input("Current ADAS13 (0 - 85)",0.0,85.0,step=0.1,value=st.session_state.ads13c,help="Current Alzheimer's Disease Assessment Scale (13 items) ")
    with columns[1]:
        ads13p = st.number_input("Previous Visit ADAS13 (0 - 85)",0.0,85.0,step=0.1,value=st.session_state.ads13p,help="Alzheimer's Disease Assessment Scale (13 items) during previous visit")
    with columns[0]:
        mmsec = st.number_input("Current MMSE (0 - 30)",0.0,30.0,step=0.1,value=st.session_state.mmsec, help="Current Mini-Mental State Examination Score")
    with columns[1]:
        mmsep = st.number_input("Previous Visit MMSE (0 - 30)",0.0,30.0,step=0.1,value=st.session_state.mmsep, help="Mini-Mental State Examination Score during previous visit")
    with columns[0]:
        raic = st.number_input("Current RAVLT_immediate (0 - 75)",0.0,75.0,step=0.1,value=st.session_state.raic, help = "Current Rey Auditory Verbal Learning Test - Immediate Recall")
    with columns[1]:
        raip = st.number_input("Previous Visit RAVLT_immediate (0 - 75)",0.0,75.0,step=0.1,value=st.session_state.raip, help = "Rey Auditory Verbal Learning Test - Immediate Recall during previous visit")
    with columns[0]:
        ralc = st.number_input("Current RAVLT_learning (-5 - 15)",-5.0,15.0,step=0.1,value=st.session_state.ralc, help="Current Rey Auditory Verbal Learning Test - Learning Score")
    with columns[1]:
        ralp = st.number_input("Previous Visit RAVLT_learning (-5 - 15)",-5.0,15.0,step=0.1,value=st.session_state.ralp, help="Rey Auditory Verbal Learning Test - Learning Score during previous visit")
    with columns[0]:
        faqc = st.number_input("Current FAQ (0 - 30)",0.0,30.0,step=0.1,value=st.session_state.faqc,help="Current Functional Activities Questionnaire Score")
    with columns[1]:
        faqp = st.number_input("Previous Visit FAQ (0 - 30)",0.0,30.0,step=0.1,value=st.session_state.faqp,help="Functional Activities Questionnaire Score during previous visit")
    with columns[0]:
        hpc = st.number_input("Current Hippocampus (2200,11300)",2200,11300,step=1,value=st.session_state.hpc,help = "Current Hippocampal volume ")
    with columns[1]:
        hpp = st.number_input("Previous Visit Hippocampus (2200,11300)",2200,11300,step=1,value=st.session_state.hpp,help = "Hippocampal volume during previous visit")
   

    

    st.session_state.agec = agec
    st.session_state.cdrc = cdrc
    st.session_state.ads11c = ads11c
    st.session_state.ads13c = ads13c
    st.session_state.mmsec = mmsec
    st.session_state.raic = raic
    st.session_state.ralc = ralc
    st.session_state.faqc = faqc
    st.session_state.hpc = hpc
    
    st.session_state.month = month
    st.session_state.cdrp = cdrp
    st.session_state.ads11p = ads11p
    st.session_state.ads13p = ads13p
    st.session_state.mmsep = mmsep
    st.session_state.raip = raip
    st.session_state.ralp = ralp
    st.session_state.faqp = faqp
    st.session_state.hpp = hpp
    
    cdrd = (cdrc - cdrp) / month
    ads11d = (ads11c - ads11p) / month
    ads13d = (ads13c - ads13p) / month
    mmsed = (mmsec - mmsep) / month
    raid = (raic - raip) / month
    rald = (ralc - ralp) / month
    faqd = (faqc - faqp) / month
    hpd = (hpc - hpp) / month
    dic2 = {'AGE_cur':agec, 'CDRSB_cur':cdrc, 'ADAS11_cur':ads11c, 'ADAS13_cur':ads13c, 'MMSE_cur':mmsec,
       'RAVLT_immediate_cur':raic, 'RAVLT_learning_cur':ralc, 'FAQ_cur':faqc,
       'Hippocampus_cur':hpc,'CDRSB_diff':cdrd, 'ADAS11_diff':ads11d, 'ADAS13_diff':ads13d, 'MMSE_diff':mmsed,
       'RAVLT_immediate_diff':raid, 'RAVLT_learning_diff':rald, 'FAQ_diff':faqd, 'Hippocampus_diff':hpd,
       }

    df2 = pd.DataFrame([dic2])
    res1 = loaded_model2.predict(df2)

    class_nam2=['Dementia', 'MCI', 'MCI to Dementia', 'NL', 'NL to MCI']

    if st.button('Predict - Progression'):
        st.session_state.button_clicked1 = not st.session_state.button_clicked1


    if st.session_state.button_clicked1:
        if res1[0] == 'Dementia':
            st.subheader('Predicted Class - DEMENTIA to DEMENTIA')
        elif res1[0] == 'NL':
            st.subheader('Predicted Class - NL- Normal to Normal')
        elif res1[0] == 'MCI':
            st.subheader('Predicted Class - MCI - Mild Cognitive Impairment to Mild Cognitive Impairment')
        elif res1[0] == 'MCI to Dementia':
            st.subheader('Predicted Class - MCI to Dementia - Mild Cognitive Impairment to DEMENTIA')
        else:
            st.subheader('Predicted Class - NL to MCI - Normal to Mild Cognitive Impairment')

        st.write(f"Probability of predicting {res1[0]} is {max(loaded_model2.predict_proba(df2)[0])} ")
        sel2 = st.radio('Choose the Explainability Principle for Progression',options=["LIME","SHAP"],)
        #st.session_state.sel2 = sel2
        if sel2 == "LIME":
            explainer = LimeTabularExplainer(X2.values, 
                                             feature_names=X2.columns, 
                                             class_names=['Dementia', 'MCI', 'MCI to Dementia', 'NL', 'NL to MCI'], random_state=np.random.seed(10))


            instance_to_explain = df2.iloc[0]

            prediction = loaded_model2.predict_proba(df2)
            predicted_class = class_nam2[np.argmax(prediction)]
            explanation = explainer.explain_instance(instance_to_explain.values, 
                                                         loaded_model2.predict_proba, 
                                                         num_features=len(X2.columns), 
                                                         top_labels=len(explainer.class_names), 
                                                          labels=[predicted_class])
            st.title('Local Interpretable Model-agnostic Explanations (LIME) for Model Prediction',
                     help="LIME is a technique used to explain the predictions of machine learning models locally. It provides insights into how the model's prediction changes for a specific instance by perturbing the input features.")
            #st.subheader(f'Predicted Class: {predicted_class}')

            # Display Lime explanation details

            for label in explanation.available_labels():
                st.subheader(f'LIME Explanation Chart for Class: {class_nam2[label]}',
                            help = "chart represents the impact of different features on the model's prediction. Positive weights indicate a feature's contribution towards the predicted class, while negative weights indicate a feature's contribution towards other classes.")
                fig = explanation.as_pyplot_figure(label)
                st.pyplot(fig)

            # Display feature importance
            st.subheader(f'LIME Feature Importance for the predicted class {predicted_class}',
                        help = "Feature importance shows the contribution of each feature to the predicted class.")
            feature_importance = explanation.as_list(label=np.argmax(prediction))
            if len(feature_importance) > 0:
                st.write(pd.DataFrame(feature_importance, columns=['Feature', 'Weight']))
            else:
                st.write("No feature importance data available for the predicted class.")



            # Display perturbation analysis
            st.subheader('LIME Perturbation Analysis')
            feature_mapping = {i: feature_name for i, feature_name in enumerate(X2.columns)}
            perturbation_data = explanation.as_map()
            for label, weights in perturbation_data.items():
                st.subheader(f'Perturbation Analysis for Class: {class_nam2[label]}',
                            help = """Perturbation analysis displays the impact of different feature values on the model's prediction.

        A positive weight suggests that an increase in the feature value contributes positively to the predicted class.
        Higher values of this feature are associated with a higher likelihood of the predicted class.

        A negative weight indicates that an increase in the feature value contributes negatively to the predicted class.
        Lower values of this feature are associated with a higher likelihood of the predicted class.

        The magnitude of the weight reflects the strength of the feature's impact on the model's prediction.
        A larger magnitude indicates a more significant influence.

        Comparing weights across different features offers insights into the relative importance of each feature for the given instance.
        """)
                #df_perturbation = pd.DataFrame(weights, columns=['Feature', 'Weight'])
                #st.write(df_perturbation)
                df_perturbation = pd.DataFrame(weights, columns=['Feature Number', 'Weight'])
                df_perturbation['Feature Name'] = df_perturbation['Feature Number'].map(feature_mapping)

                # Display the DataFrame with feature names
                st.write(df_perturbation[['Feature Name', 'Weight']])
        else:

            st.title("SHapley Additive exPlanations (SHAP)",help = """SHAP values aim to fairly distribute the contribution of each feature to the model's prediction across all possible combinations of features. """)
                # SHAP Explanation
            shexplainer = shap.Explainer(loaded_model2)
            shap_values = shexplainer.shap_values(df2)
            np.random.seed(10)


            # Summary Plot
            st.subheader('Summary Plot', help = """The SHAP summary plot provides an overview of feature importance for each class.
            It displays the magnitude and direction of the impact of each feature on the model's output.
            Features with positive SHAP values contribute positively to the predicted class and negative SHAP values contribute 
            negatively to the predicted class.""")
            fig_summary = shap.summary_plot(shap_values, df2, show=False,class_names=class_nam2)
            st.pyplot(fig_summary)



            # Force Plot
            st.subheader('Force Plot',help = """Force plot shows how each feature contributes to the prediction of the 
            selected class for a given instance. Features with positive contributions increase the likelihood of the 
            predicted class and features with negative contributions decrease the likelihood of the predicted class.
        Red bars represent higher feature values, while blue bars represent lower feature values.""")
            i=0
            for f in class_nam2:
                st.write(f"Features contributed for the class {f} ")
                fp = shap.plots.force(shexplainer.expected_value[i], shap_values[i], df2, matplotlib = True)
                st.pyplot(fp)
                i+=1

            # Decision Plot
            st.subheader("Decision Plot",help = """Decision plot shows the decision path of the model for a given instance, highlighting the contributions of different features.

        Feature Contributions: Observe how different features influence the decision at various points.

        Positive and Negative Contributions: Positive contributions increase the likelihood of the predicted class, while negative contributions decrease it.""")
            dp = shap.decision_plot(shexplainer.expected_value[1], shap_values[1], df2.columns)
            st.pyplot(dp)
    

  

 # main
def main():
    st.sidebar.title("Explore Alzheimer's Disease Prediction System",help = "1. To predict the current status with the current data choose Predict - Explain Current\n 2. To predict the progression and if you have the current and past visit data choose Predict - Explain Progression")
    pages = ["Home", "Predict - Explain Current", "Predict - Explain Progression"]

    if 'active_page' not in st.session_state:
        st.session_state.active_page = pages[0]

    if 'age' not in st.session_state:
        st.session_state.age = 50
    if 'month' not in st.session_state:
        st.session_state.month = 1
    if 'agec' not in st.session_state:
        st.session_state.agec = 50
    if 'cdr' not in st.session_state:
        st.session_state.cdr = 0.0
    if 'cdrp' not in st.session_state:
        st.session_state.cdrp = 0.0
    if 'cdrc' not in st.session_state:
        st.session_state.cdrc = 0.0
    if 'ads11' not in st.session_state:
        st.session_state.ads11 = 0.0
    if 'ads11c' not in st.session_state:
        st.session_state.ads11c = 0.0
    if 'ads11p' not in st.session_state:
        st.session_state.ads11p = 0.0
    if 'ads13' not in st.session_state:
        st.session_state.ads13 = 0.0  
    if 'ads13p' not in st.session_state:
        st.session_state.ads13p = 0.0  
    if 'ads13c' not in st.session_state:
        st.session_state.ads13c = 0.0  
    if 'mmse' not in st.session_state:
        st.session_state.mmse = 0.0   
    if 'mmsep' not in st.session_state:
        st.session_state.mmsep = 0.0
    if 'mmsec' not in st.session_state:
        st.session_state.mmsec = 0.0  
    if 'rai' not in st.session_state:
        st.session_state.rai = 0.0 
    if 'raic' not in st.session_state:
        st.session_state.raic = 0.0 
    if 'raip' not in st.session_state:
        st.session_state.raip = 0.0 
    if 'ral' not in st.session_state:
        st.session_state.ral = 0.0 
    if 'ralc' not in st.session_state:
        st.session_state.ralc = 0.0
    if 'ralp' not in st.session_state:
        st.session_state.ralp = 0.0
    if 'faq' not in st.session_state:
        st.session_state.faq = 0.0 
    if 'faqc' not in st.session_state:
        st.session_state.faqc = 0.0  
    if 'faqp' not in st.session_state:
        st.session_state.faqp = 0.0  
    if 'hp' not in st.session_state:
        st.session_state.hp = 2200  
    if 'hpc' not in st.session_state:
        st.session_state.hpc = 2200
    if 'hpp' not in st.session_state:
        st.session_state.hpp = 2200

    st.session_state.active_page = st.sidebar.radio("Select the page to navigate", pages, index=pages.index(st.session_state.active_page))   
    
    if st.session_state.active_page == "Home":
        home()
    elif st.session_state.active_page == "Predict - Explain Current":
        predict_explain_current()
    elif st.session_state.active_page == "Predict - Explain Progression":
        predict_explain_progression()


if __name__ == "__main__":
    main()
