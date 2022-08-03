# Streamlit footer addition and remove the default streamlit features
HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
footer:after {
    content:'copyrights © 2021 rahul.kumeriya@outlook.com\
        [ DONT FORGET TO PLANT TREES ]' ;
    visibility: visible;
    display: block;
    position: fixed;
    #background-color: red;
    padding: 1px;
    bottom: 0;
}
</style>"""

# Application Formatting for good display
PADDING = 0
MAIN_STYLE = """ <style>
    .reportview-container .main .block-container{{
        padding-top: {PADDING}rem;
        padding-right: {PADDING}rem;
        padding-left: {PADDING}rem;
        padding-bottom: {PADDING}rem;
    }} </style> """

SideBar_Style = """ <style>
.css-hxt7ib {{
    padding-top: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}} </style> """