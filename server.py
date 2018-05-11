from flask import Flask, render_template, request, make_response
from predict import main

app = Flask(__name__)

x = list()
y = list()
z = list()

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
    global x, y, z
    file = request.files['dataset']
    interval = request.form['interval']

    destination = 'static/dataset.csv'
    file.save(destination)

    linear, rbf, date, a1, a2, coef, const = main(str(interval),destination)
    x = date
    y = linear
    z = rbf
    return render_template('result.html', valv = [linear, rbf, date, a1, a2, coef, const, len(linear)])

@app.route('/plot.png')
def plot():
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig = Figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot_date(x, y, '-',color='green')
    ax2.plot_date(x, z, '-',color='blue')
    ax1.grid()
    ax2.grid()
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Linear')
    ax2.set_title('RBF')

    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = io.BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response


if __name__ == '__main__':
   app.run(debug=True)
