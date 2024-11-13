import streamlit as st
import pickle
import numpy as np
import os 

# تحميل النموذج من المجلد
def load_model():
    model_path = "house_price_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    else:
        st.error("لم يتم العثور على ملف النموذج. تأكد من أن المسار صحيح.")
        return None



def main():
    '''UI CODE : Streamlit code'''
    st.title("House Price Regression ML")
    
    # تحميل النموذج 
    model = load_model()
    
    with st.form(key='input_form'):
        Square_Footage = st.number_input("Square Footage",step=100)#
        Num_Bedrooms = st.selectbox("Bedrooms",[1,2,3,4,5])
        Num_Bathrooms = st.selectbox("Bathrooms",[1,2,3])     
        Year_Built = st.number_input("Year Built", min_value=1950, max_value=2022,step=1) 
        Lot_Size = st.number_input("Lot Size",min_value=0.51, max_value=4.99,step=0.01)
        Garage_Size = st.selectbox("Garage Size",[0,1,2])
        Neighborhood_Quality = st.selectbox("Neighborhood Quality",[1,2,3,4,5,6,7,8,9,10])
        
        
        submit_button = st.form_submit_button("Predict")
        
        if submit_button:
            features = np.array([Square_Footage, Num_Bedrooms, Num_Bathrooms, Year_Built, Lot_Size, Garage_Size, Neighborhood_Quality])
            prediction = model.predict([features])
            st.write(f"The predicted price of the house is ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()