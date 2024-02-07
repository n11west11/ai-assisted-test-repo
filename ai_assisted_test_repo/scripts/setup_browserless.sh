#v1
docker run -p 3000:3000 -p 5900:5900 --name browserless_v1 -e "DEMO_MODE=true" -e "MAX_CONCURRENT_SESSIONS=15" -e CONNECTION_TIMEOUT=5400000 -e PWDEBUG=1 browserless/chrome:1-arm64 
# v2
docker run -p 3000:3000 -p 5900:5900 --name browserless -e "DEFAULT_LAUNCH_ARGS=[\"--window-size=1920,1080\"]" -e "DEMO_MODE=true" -e "CONCURRENT=15" -e TIMEOUT=5400000 -e PWDEBUG=1 ghcr.io/browserless/chrome

