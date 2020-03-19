from flask import Flask, escape, request, render_template, send_file, redirect, make_response, send_from_directory, flash
from npa import *
# reset()
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def hello_world():
    return redirect("/appUpload")

@app.route('/appUpload')
def appUpload():
    loan_file = get_stats()
    return render_template('app_upload.html', loan_file = loan_file)

@app.route('/appUpdate', methods=['POST', 'GET'])
def appUpdate():
    file = request.files['fileupload']
    file1 = request.files['fileupload1']
    file2 = request.files['fileupload2']

    file.save('./dataset/appraisal.csv')
    file1.save('./dataset/application.csv')
    file2.save('./dataset/loan_performance.csv')
    if len(os.listdir('./model1')):
        os.remove('./model1/model.pkl')
    if len(os.listdir('./model2')):
        os.remove('./model2/model.pkl')

    save_app_data('./dataset/appraisal.csv','./dataset/application.csv','./dataset/loan_performance.csv')
    return redirect("/appUpload")

@app.route('/npaUpload')
def npaUpload():
    return render_template('npa_upload.html')

@app.route('/npaUpdate', methods=['POST', 'GET'])
def npaUpdate():
    file = request.files['fileupload']
    # print(file)
    file.save('./temp_.csv')
    update_default_data('./temp_.csv')
    return redirect("/appUpload")

@app.route('/getRisk')
def getRisk():
    return render_template("get_risk.html")

@app.route('/computeRisk', methods=['POST', 'GET'])
def computeRisk():
    file = request.files['fileupload']
    df = pd.read_csv('./newApplicant.csv')
    X,flag = return_applicant(df)
    scores = []
    for i in X:
        if flag == 0:
            loaded_model = pickle.load(open('./model2/model.pkl', 'rb'))
        else:
            loaded_model = pickle.load(open('./model1/model.pkl', 'rb'))

        score = loaded_model.predict_proba(i.reshape(1,-1))
        scores.append(max_value(score[0]))
        flash("Risk score for the applicant is: " + str(max_value(score[0])))
    
    ids = df['ApplicationId']
    data = {'ApplicationID':ids, 'Risk Scores':scores} 
  
    df = pd.DataFrame(data)
    df.to_csv('riskScores.csv',index=False)
    return redirect("/getRisk")

@app.route('/reset')
def reset_():
    reset()
    return redirect("/appUpload")


if __name__ == '__main__':
    app.run(debug=True)
