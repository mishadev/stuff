DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker run -it -v "$DIR/scripts:/scripts" scikit-learn