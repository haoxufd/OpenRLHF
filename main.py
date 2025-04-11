# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-f628a3c417754f0aaeb4137447da7fa0", base_url="https://api.deepseek.com")

system_message = """
You are a senior mathematics professor assessing a student's multi-step solution. Your evaluation will be based on the following inputs:
The original mathematical question
A reference solution (for comparison)
A sequence of previously verified correct steps from the student's answer
A new step to evaluate

Evaluation Criteria:
Your primary task is to determine whether the new step is CORRECT or INCORRECT. If the new step (combined with prior steps) follows the same logical progression as the reference solution, it is correct. However, not following the path of reference answer doesn't mean it is incorrect, since the reference solution is not the only valid approach. A step may deviate from the reference yet still be correct if it:
Follows logically from the previous steps,
Adheres to mathematical principles, and
Contributes soundly toward solving the problem.
"""

user_content_1 = "<question>Ahmed and Emily are having a contest to see who can get the best grade in the class. There have been 9 assignments and Ahmed has a 91 in the class. Emily has a 92. The final assignment is worth the same amount as all the other assignments. Emily got a 90 on the final assignment. What is the minimum grade Ahmed needs to get to beat Emily if all grades are whole numbers?</question><reference_solution>Ahmed has scored 819 total points in the class thus far because 9 x 91 = <<9*91=819>>819\nEmily had scored 828 total points before the final assignments because 9 x 92 = <<9*92=828>>828\nShe scored 918 total points after the final assignment because 828 + 90 = <<828+90=918>>918\nAhmed needs to score a 99 to tie Emily for the semester because 918 - 819 = <<918-819=99>>99\nAhmed needs to score a 100 to beat Emily for the semester because 99 + 1 = <<99+1=100>>100\n#### 100</reference_solution><previous_steps></previous_steps><step_to_evaluate>Ahmed needs to beat Emily's class average, which is 92 - 9 = <<92-9=83>>83.</step_to_evaluate>"

user_content_2 = "<question>A man is purchasing a pair of sneakers at a club store where he receives a membership discount of 10% off any purchase.  In addition to the membership discount, the man also has a coupon for $10 off any pair of sneakers.  If the man wants to purchase a $120 pair of sneakers and the membership discount must be applied after the coupon, how much will he pay for the sneakers after using both the coupon and membership discount?</question><reference_solution>The coupon is applied first, which reduces the cost of the sneakers to 120 - 10 = $<<120-10=110>>110.\nNext, the 10% discount leads to a reduction in price of 110*0.1 = $<<110*0.1=11>>11.\nTherefore, the final price the man must pay is 110 - 11 = $<<110-11=99>>99.\n#### 99</reference_solution><previous_steps>The man has a $10 coupon off of any pair of sneakers, so the cost of the sneakers is reduced to 120 - 10 = $<<120-10=110>>110.\n<|reserved_special_token_0|>The man also has a 10% discount off of this reduced price, so his discount is worth 110 × 10/100 = $<<110*10/100=11>>11.\n</previous_steps><step_to_evaluate>The reduced price of the sneakers with the coupon applied can not be further reduced by the membership discount, which would make its amount a negative number. Therefore, before this could be done, another price reduction needs to be applied in order to ensure this does not occur.\n</step_to_evaluate>"

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content_2},
    ],
    stream=False
)

print(response.choices[0].message.content)