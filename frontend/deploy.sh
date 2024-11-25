read_var() {
  if [ -z "$1" ]; then
    echo "environment variable name is required"
    return 1
  fi

  local ENV_FILE='.env'
  if [ ! -z "$2" ]; then
    ENV_FILE="$2"
  fi

  local VAR=$(grep $1 "$ENV_FILE" | xargs)
  IFS="=" read -ra VAR <<< "$VAR"
  echo ${VAR[1]}
}

endpoint=$(read_var REGISTRY)
cat ./.env
npm install
npm run build
echo "Building and pushing dockerfile to $endpoint..."
docker build -t $endpoint . && docker push $endpoint
rm -R ./dist

sleep 2
echo "=====Done===="
