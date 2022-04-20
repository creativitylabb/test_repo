# Modified from
# https://github.com/MaartenGr/streamlit_guide/blob/master/Procfile
# and
# https://github.com/MaartenGr/streamlit_guide/blob/master/setup.sh

mkdir -p ~/.streamlit/

echo "\
[theme]\n\
base = dark\n\
primaryColor=#07666b\n\
backgroundColor=#0E1117\n\
secondaryBackgroundColor=#262730\n\
textColor=FAFAFA
[server]\n\
headless = true\n\
maxMessageSize= 500\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml