import streamlit as st
import helper
import pickle

model = pickle.load(open('model.pkl','rb'))

# to place image in center

st.image('duplicatedetect.png', width=700)
# st.image('duplicatedetect.png', width = 700)

st.title('Duplicate Detection')
# st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    if(q1 == '' or q2 == ''):
        st.warning('Please enter both questions')
        # st.stop()
    else:
        result = model.predict(query)[0]
        st.success('Prediction Successful')
        if result:
            st.markdown('#### Duplicate')
        else:
            st.markdown('#### Not Duplicate')


