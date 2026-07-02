### Comando completo para descargar dependencias y ejecutar tests

cd C:\Users\vicen\Dropbox\Escritorio\Nacho-UNI\TFG\AmITraceable\users
rmdir /s /q venv
python -m venv C:\venvs\amitraceable-users
C:\venvs\amitraceable-users\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m pytest --cov=app --cov-report=xml --cov-report=term-missing


### Advertencia Silenciada: from click.parser import split_arg_string
Ese warning es totalmente inofensivo: no proviene de mi código, es una incompatibilidad menor entre spaCy y la versión de click que arrastra como dependencia (spaCy usa una API interna de click que va a moverse de sitio en su versión 9.0, y click avisa con antelación). No afecta a nada de lo que hace tu tool: es solo una advertencia de que en el futuro, cuando salga Click 9.0, spaCy tendrá que actualizar ese import.
Ha sido silenciada en pyproject.toml