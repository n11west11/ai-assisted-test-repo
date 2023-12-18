docker run -p 3000:3000 -p 5900:5900 --name browserless -e "DEBUG=true" -e "ENABLE_DEBUGGER=true" -e "MAX_CONCURRENT_SESSIONS=10" -e CONNECTION_TIMEOUT=5400000 -e PWDEBUG=1 browserless/chrome
