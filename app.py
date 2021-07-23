from flask import Flask, render_template, redirect
from forms import PlagForm
import Requests_ex
app = Flask(__name__)
app.config['SECRET_KEY'] = '827e763e55e41e32381c13afc4338d5f'
get_text = []

@app.route('/index')
@app.route('/', methods=['POST', 'GET'])
def index():
    form = PlagForm()
    if form.validate_on_submit():
        text = form.checkText.data
        get_text.clear()
        get_text.append(text)
        return redirect('/results')
    return render_template('index.html', form=form)

@app.route('/results')
def results():
    ans = Requests_ex.main_function(get_text[0])
    return render_template('results.html', output=ans)

if __name__ == '__main__':
    app.run()
    