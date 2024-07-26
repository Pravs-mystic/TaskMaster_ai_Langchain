import streamlit as st
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import SequentialChain

# Initialize the LLM (e.g., OpenAI's GPT-3/4)
llm = OpenAI(temperature = 0.6)


# Define the system prompt
system_prompt = "You are an intelligent assistant helping the user to plan their daily tasks. Your responses should be helpful, very concise, and motivational."

# Define the chains with the system prompt included
def process_tasks(tasks_input):
    tasks_list = [task.strip() for task in tasks_input.split(',')]
    return tasks_list


prompt_template_prioritize = PromptTemplate(
    input_variables=["daily_tasks"],
    template=f"{system_prompt}\nTake the following list of tasks: {{daily_tasks}} and prioritize them into High, Medium, and Low categories.Use the format '''daily_tasks1:priority_type,daily_tasks2:priority_type,...''' for each task."
)
prioritize_chain = LLMChain(llm=llm, prompt=prompt_template_prioritize, output_key="prioritized_tasks")

prompt_template_time_estimate = PromptTemplate(
    input_variables=["daily_tasks"],
    template=f"{system_prompt}\nEstimate the time required to complete each task in the list: {{daily_tasks}}. Provide the estimates in hours or minutes which is more appropriate. Use the format '''daily_tasks1:Time_required,daily_tasks2:Time_required,...''' for each task."
)
time_estimate_chain = LLMChain(llm=llm, prompt=prompt_template_time_estimate, output_key="time_estimates")

prompt_template_breaks = PromptTemplate(
    input_variables=["time_estimates"],
    template=f"{system_prompt}\nBased on the following time estimates: {{time_estimates}}, suggest appropriate break intervals to maintain productivity.  Use the format '''After daily_tasks1:Break_time_required,After daily_tasks2:Break_time_required,...''' for each task."
)
breaks_chain = LLMChain(llm=llm, prompt=prompt_template_breaks, output_key="break_recommendations")

prompt_template_quotes = PromptTemplate(
    input_variables=["daily_tasks"],
    template=f"{system_prompt}\nFor each task in {{daily_tasks}}, provide a motivational quote to inspire the user to complete the task.  Use the format '''daily_tasks1:quote,daily_tasks2:quote,...''' for each task."
)
quotes_chain = LLMChain(llm=llm, prompt=prompt_template_quotes, output_key="motivational_quotes")


# Chain the components together
daily_planner_chain = SequentialChain(
    chains=[prioritize_chain, time_estimate_chain, breaks_chain, quotes_chain],
    input_variables=["daily_tasks"],
    output_variables=["prioritized_tasks", "time_estimates", "break_recommendations", "motivational_quotes"]
)

# Streamlit App
st.title("Your Daily Task Planner")

tasks_input = st.text_area("Enter your tasks for the day as comma separated values:", "")

if st.button("Generate Plan"):
    daily_tasks = process_tasks(tasks_input)
    inputs = {'daily_tasks': daily_tasks}
    final_output = daily_planner_chain(inputs)
    
    st.subheader("Daily Tasks")
    st.write(final_output["daily_tasks"])
    
    st.subheader("Prioritized Tasks")
    st.write(final_output["prioritized_tasks"])
    
    st.subheader("Time Estimates")
    st.write(final_output["time_estimates"])
    
    st.subheader("Break Recommendations")
    st.write(final_output["break_recommendations"])
    
    st.subheader("Motivational Quotes")
    st.write(final_output["motivational_quotes"])
    
