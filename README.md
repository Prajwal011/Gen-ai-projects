# To run py files use below code:


from pyngrok import ngrok

#### Set authentication token if you haven't already done so
ngrok.set_auth_token('ngork auth token')

#### Start Streamlit server on a specific port
!nohup streamlit run st.py --server.port 5011 &

#### Start ngrok tunnel to expose the Streamlit server
ngrok_tunnel = ngrok.connect(addr='5011', proto='http', bind_tls=True)

#### Print the URL of the ngrok tunnel
print(' * Tunnel URL:', ngrok_tunnel.public_url)
