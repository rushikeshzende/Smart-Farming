from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename


from tensorflow.keras.models import load_model, Sequential  # Build NN
from tensorflow.keras.layers import Dense  # creating layers

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_crop')
def predict_crop():
    return render_template('crop_prediction.html')

@app.route('/nutrition_deficiency', methods=['GET', 'POST'])
def nutrition_deficiency():
    return render_template('nutrition_deficiency.html')


@app.route('/predict_crop_price')
def predict_crop_price():
    return render_template('crop_price_prediction.html')


def prediction_model(data):
     model = load_model('crop_prediction.h5')#load model
     result= model.predict(data)#predict on data
     return result


@app.route('/predict_crop_ans', methods=['POST'])
def submit_form():
    # Retrieve form data
    
    state_list=['andaman and nicobar islands','andhra pradesh','arunachal pradesh','assam','bihar','chandigarh', 'chhattisgarh','dadra and nagar haveli','goa','gujarat','haryana','himachal pradesh','jammu and kashmir','jharkhand','karnataka','kerala','madhya pradesh','maharashtra','manipur','meghalaya','mizoram','nagaland','odisha','puducherry','punjab','rajasthan','sikkim','tamil nadu','telangana', 'tripura','uttar pradesh','uttarakhand','west bengal']# aplha
    Crop= ['apple','arecanut','ashgourd','banana','barley','beetroot','bittergourd','blackgram','blackpepper','bottlegourd','brinjal','cabbage','cardamom','carrot','cashewnuts','cauliflower','coffee','coriander','cotton','cucumber','drumstick','garlic','ginger','grapes', 'horsegram','jackfruit','jowar','jute','ladyfinger','maize','mango', 'moong','onion','orange','papaya','pineapple','pomegranate','potato','pumpkin','radish','ragi','rapeseed','rice','ridgegourd','sesamum','soyabean','sunflower','sweetpotato','tapioca','tomato','turmeric','watermelon','wheat']
    state_name =request.form.get('state_name').lower()
    nitrogen = float(request.form.get('nitrogen'))
    phosphorus = float(request.form.get('phosphorus'))
    potassium = float(request.form.get('potassium'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))
    temperature = float(request.form.get('temperature'))
    
    data=np.array([[state_list.index(state_name),nitrogen,phosphorus,potassium,ph,rainfall,temperature]])
    result = prediction_model(data)
    
     # Get predicted class
    predicted_class_index = np.argmax(result, axis=1)[0]
    predicted_label = Crop[predicted_class_index]
    
    # render the template with the result
    return render_template("display.html",data=predicted_label)
    
# crop pice detection backend section
def crop_price_prediction_model(data):
     model = load_model('crop_price_prediction_model1.h5')#load model
     result= model.predict(data)#predict on data
     return result

@app.route('/predict_crop_price_ans', methods=['POST'])
def submit_form_crop_price_pred():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    State_name = ['Karnataka', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']
    City_name = ['Amritsar', 'Bangalore', 'Belgaum', 'Bellary', 'Chennai', 'Coimbatore', 'Davanagere', 'Durgapur', 'Gulbarga', 'Hubli', 'Kanpur', 'Kolkata', 'Lucknow', 'Ludhiana', 'Madurai', 'Mangalore', 'Mumbai', 'Mysore', 'Nagpur', 'Patiala', 'Pune', 'Shimoga', 'Siliguri', 'Udupi', 'Varanasi']	
    Crop_Type = ['Barley', 'Coffee', 'Cotton', 'Groundnut', 'Jute', 'Maize', 'Millets', 'Mustard', 'Onion', 'Potato', 'Pulses', 'Rice', 'Sesame', 'Soybean', 'Sugarcane', 'Sunflower', 'Tea', 'Tobacco', 'Tomato', 'Wheat']
    Season =['Kharif', 'Post-Monsoon', 'Rabi', 'Zaid']
    month=request.form.get('month')
    state=request.form.get('state')   
    city=request.form.get('city')
    crop_type=request.form.get('crop_type')
    season=request.form.get('season')
    temperature=float(request.form.get('temperature'))
    rainfall=float(request.form.get('rainfall'))
    supply_volume=float(request.form.get('supply_volume'))
    demand_volume=float(request.form.get('demand_volume'))
    transportation_cost=float(request.form.get('transportation_cost'))
    fertilizer_usage=float(request.form.get('fertilizer_usage'))
    pest_infestation=float(request.form.get('pest_infestation'))
    market_competition=float(request.form.get('market_competition'))

    data=np.array([[month.index(month),State_name.index(state),City_name.index(city),Crop_Type.index(crop_type),Season.index(season),temperature,rainfall,supply_volume,demand_volume,transportation_cost,fertilizer_usage,pest_infestation,market_competition]])
    result = crop_price_prediction_model(data)

    # Get predicted class
    month_re=months[int(month)]
    return render_template("/crop_price_display.html",data=result[0][0], crop_type=crop_type, mo=month_re)


    
                            ### NUTRTION DEFICIENCY PREDICTION ###
                               
# Configurations for file uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/nutrition_deficiency_prediction', methods=['POST'])
def nutrition_deficiency_prediction_submit_form():
    file = request.files['image']

    if file and allowed_file(file.filename):
        # Secure the filename and save it in the uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the file permanently
        
        # Generate the URL to display the image
        image_path = url_for('static', filename=f'uploads/{filename}')

        
        # Load the trained model
        model = load_model('nutrition_deficeincy_pred.h5')  # Updated model path

        # Path to your custom image
        image_path = url_for('static', filename=f'uploads/{filename}')# Replace with your image path

        # Preprocess the image
        IMG_SIZE = (224, 224)  # Must match the size used during training

        def preprocess_image(image_path):
            img = load_img(image_path, target_size=IMG_SIZE)  # Load and resize image
            img_array = img_to_array(img)  # Convert image to array
            img_array = img_array / 255.0  # Rescale to 0-1 range
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        
        image = preprocess_image(file_path)

        # Make prediction
        prediction = model.predict(image)
        print("the probabilities is",prediction)

        # Class labels (ensure they match the model's training)
        class_labels = ['Iron_Deficiency', 'Magnisium_Deficiency', 'Nitrogan_Deficiency','Phosphorus_Deficiency','Potasium_Deficiency']
        
        # Get the 3 predicted classes using index which is more related
        class_prediction_dict = dict(zip(class_labels, prediction[0]))
        

        sorted_class_prediction_dict = sorted(class_prediction_dict.items(), key=lambda x: x[1], reverse=True)
      

        return render_template('/nutrition_deficiency_prediction_dislpay.html', image_path=image_path, 
                               class1=sorted_class_prediction_dict[0][0], 
                               class2 = sorted_class_prediction_dict[1][0], 
                               class3=sorted_class_prediction_dict[2][0] )
        
        

    
     # Delete the uploaded file after rendering
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
    else:
        pass
        

    return 'File not allowed', 400









if __name__ == '__main__':
    app.run(debug=True)
