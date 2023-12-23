import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Sample keywords for success and failure
success_keywords = [
    'innovative', 'cutting-edge', 'market leader', 'high growth', 'strong customer base',
    'customer satisfaction', 'positive cash flow', 'efficient operations', 'expansion',
    'industry recognition', 'award-winning', 'disruptive technology', 'strategic partnerships',
    'profitable', 'innovative solution', 'market demand', 'global presence', 'customer loyalty',
    'scalable model', 'unique value proposition', 'experienced team', 'positive reviews',
    'consistent revenue growth', 'leading in market share', 'highly skilled workforce', "Innovation", "Market Validation", "Customer- 
    Centric Approach", "Agile Execution", "Adaptability", "Talent Acquisition", "Strategic Partnerships", "Financial Management",             "Product-Market Fit", "Scalability", "User Experience", "Effective Communication", "Iterative Development", "Brand Building",             "Problem Solving", "Risk Management", "Customer Feedback", "Continuous Learning", "Resilience", "Time Management", "Networking",          "Data-Driven Decision Making", "Focus on Core Competencies", "Community Engagement", "Diversity and Inclusion", "Marketing Strategy", "Regulatory Compliance", "Flexibility", "Strategic Vision", "Cost-Efficiency", "Rapid Prototyping", "Leadership", "Embracing Failure", "Agile Culture", "Adoption of Technology", "Sustainable Practices", "Customer Retention", "Aggressive Marketing", "Competitive Analysis", "Resource Optimization", "Informed Decision-Making", "Lean Operations", "Investor Relations", "Global Expansion", "Ecosystem Collaboration", "Intrapreneurship", "Market Disruption", "Brand Authenticity", "Digital Transformation", "Data Security", "Lean Startup Methodology", "Strategic Vision", "Hiring Agility", "Cultural Diversity", "Intellectual Property Protection", "Adaptive Leadership", "User Retention", "Strategic Alliances", "Mobile Responsiveness", "Continuous Innovation", "Holistic Customer Experience", "Open Communication Channels", "Agile Product Development", "Community Building", "Product Differentiation", "Innovative Marketing Campaigns", "Scalable Infrastructure", "Operational Efficiency", "Smart Resource Allocation", "Funding Diversity", "Leveraging Social Media", "Holacracy", "Corporate Social Responsibility", "Market Trends Awareness", "Transparent Governance", "Iterative Prototyping", "Pivot Capability", "Learning Organization", "Brand Loyalty", "Real-Time Analytics", "Inclusive Leadership", "Strategic Pivot", "Strategic Exit Planning", "Blockchain Integration", "Crowdsourcing Wisdom", "Customer-Centric Innovation", "Platform Thinking", "Agile Marketing", "Human-Centered Design", "Global Mindset", "Predictive Analytics", "Strategic Storytelling", "Subscription-Based Revenue Models", "Blockchain Integration", "Holistic Employee Wellbeing", "Predictive Analytics", "Strategic Storytelling", "Subscription-Based Revenue Models", "Blockchain Integration", "Holistic Employee Wellbeing"
]

failure_keywords = [
    'struggling', 'financial difficulties', 'poor management', 'bankruptcy', 'limited resources', 'failing to collaborate with vendors',
    'low customer retention', 'negative cash flow', 'lack of direction', 'declining market share',
    'unstable leadership', 'poor product-market fit', 'low customer satisfaction', 'inefficient operations',
    'lack of innovation', 'market rejection', 'poor sales performance', 'bad reviews',
    'declining user engagement', 'legal issues', 'poor execution', 'high turnover rate',
    'stagnant growth', 'revenue decline', 'negative public perception', "Lack of Product-Market Fit", "Poor Financial Management", "Inadequate Market Research", "Resistance to Change", "Insufficient Adaptability", "Talent Shortage", "Failed Partnerships", "Lack of Strategic Vision", "Ineffective Marketing", "Slow Execution", "Poor User Experience", "Communication Breakdown", "Lack of Innovation", "Weak Branding", "Inability to Solve Problems", "Ignoring Risk Management", "Ignoring Customer Feedback", "Lack of Learning", "Lack of Resilience", "Poor Time Management", "Isolation", "Data Neglect", "Core Competency Neglect", "Community Disengagement", "Lack of Diversity and Inclusion", "Strategic Mistakes", "Non-compliance with Regulations", "Inflexibility", "Lack of Visionary Leadership", "Fear of Failure", "Lack of Agile Culture", "Failure to Adopt Technology", "Unsustainable Practices", "Customer Dissatisfaction", "Weak Marketing Strategy", "Ignoring Competitive Analysis", "Resource Mismanagement", "Lack of Informed Decision-Making", "Inefficiency in Operations", "Poor Investor Relations", "Failed Global Expansion", "Lack of Collaboration", "Intrapreneurship Failure", "Resistance to Digital Transformation", "Data Security Breaches", "Failure to Grasp Market Trends", "Ineffective Crisis Management", "Failure in Growth Hacking", "Unempowered Customers", "Storytelling Failure", "Ineffective Supply Chains", "Extravagant Spending", "Lack of Cross-Functional Collaboration", "Employee Wellbeing Neglect", "Holacracy Failure", "Neglecting Corporate Social Responsibility", "Overlooking Intellectual Property Protection", "Poor Brand Authenticity", "Failure to Pivot Strategically", "Ignoring Blockchain Integration", "Neglecting Employee Diversity", "Failing to Predict Analytics", "Ignoring Subscription-Based Revenue Models", "Ignoring Lean Startup Methodology", "Neglecting Sustainable Supply Chains", "Ignoring Predictive Analytics", "Neglecting Holistic Employee Wellbeing"
]

# Generate synthetic dataset
data = {
    'description': [
        'innovative product with cutting-edge technology',
        'struggling to gain market traction',
        'successful startup with a strong customer base',
        'facing financial difficulties',
        'disruptive solution for industry challenges',
        'poor management and lack of direction',
        'rapid growth and expansion',
        'declining user engagement',
        'revolutionary idea but limited resources',
        'established market leader with consistent growth',
        'failing to collaborate with vendors'
    ] * 200,  # Repeat the descriptions to create a larger dataset
    'success_label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 200  # 0: Failure, 1: Success
}

df = pd.DataFrame(data)

# Create features based on the presence of keywords
df['success_feature'] = df['description'].apply(lambda x: any(keyword in x for keyword in success_keywords))
df['failure_feature'] = df['description'].apply(lambda x: any(keyword in x for keyword in failure_keywords))

# Assuming 'success_label' is the target variable
features = df[['success_feature', 'failure_feature']]
target = df['success_label']

# Initialize a RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model
model.fit(features, target)

# Streamlit app
st.title("Startup Success Prediction")

# User input for startup features
user_input = st.text_area("Enter a description of your startup:")

if st.button("Predict"):
    # Convert user input to numerical features using the model's keywords
    user_features = pd.DataFrame({
        'success_feature': [any(keyword in user_input for keyword in success_keywords)],
        'failure_feature': [any(keyword in user_input for keyword in failure_keywords)]
    })

    # Make prediction for user input
    prediction = model.predict(user_features)

    # Display prediction
    if prediction[0] == 1:
        st.success("The model predicts that your startup is likely to succeed. Good luck! for a start to break bounds of everyone's imagination.")
    else:
        st.error("The model predicts that your startup is likely to fail. Consider providing more appealing features to customers and take up the universe.")
