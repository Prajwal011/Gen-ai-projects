# To run py files in colab use below code:


from pyngrok import ngrok

ngrok.set_auth_token('ngork auth token')

!nohup streamlit run filename.py --server.port 5011 &

ngrok_tunnel = ngrok.connect(addr='5011', proto='http', bind_tls=True)

print(' * Tunnel URL:', ngrok_tunnel.public_url)
