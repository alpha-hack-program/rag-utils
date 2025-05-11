# Intro with the title slide

<!-- Nice title, isn't it? -->

So Click, Push, Deploy and you have a ChatBot you own, for yourself... easy right?

Today I want to do two things, one is deploying a ChatBot from the scratch and the other is explain why and how.

Because today I have little time to do it...(thanks team!) I will have to do it in paralell.

So, let's start with the demo...

# Demo part 1

I repeat around 500 times a week the four pillars of MLOps or DevOps for AI projects... Flows Automation Monitoring and Experimientation.

This is a good opportunity to execise the Automation pillar.

We're going to deploy our chatbot using RHOAI and GitOps, let's go to the console.

As you can see there's no project called `doc-bot`, if every goes as planned in a few minutes we'll have everything we need, namely:
- a vector database, more on this later
- an LLM (or I should say a small language model as my colleague Myriam has explained before) :wink:
- a couple of chatbot apps, why one when you can have multiple!
- all in a project call `doc-bot`

Well, those alergyc to the command line, please close your eyes... it won't be long.

> ./deploy.sh

Done, come on, those alergycs... open your eyes.

Now... let's see quickly if it's progressing as expected and hope for the best.

# Back to the slides

Let me start from the beginning... this all started with a requirement by a customer some months ago.

> Slide: I want to talk to my documents

The proposal was: "I want to talk to my documents".

Let me explain this a bit, the customer is a local government and they want to expedite the resolution of processes. But as we all know these processes are complex and involve reading complex documentation. So we have to...

> Slide: Help the civil servant extracting information to make an informed decision

I started shaping the solution as I was listening to the customer... this is a RAG, 100%... come on it has to be it!


> Slide: RAG

```
    /-- input query --> GenAI App -- search relevant information -> Vector DB
   |                              \-- input query + enhanced context -> LLM
User <--------------- response  --/
```


And why RAG? What on Earth is RAG?


Maybe some of you know, but for all the rest. RAG means Retrival Augmented Generation... I know, it sounds a bit weird for a Spaniard... and maybe to the Portuguese, too.

The concept is simple. Have you used ChatGPT to ask questions, summarize of whatever around a PDF you just uploaded?

> Slide: ChatGPT + PDF

Well, this means that you can pass a context, the PDF, to an LLM along with the question or task you want it to do.

It is, as a friend says, going to the exam with the book.

That's fine but... when you deploy an LLM on you datacenters is more a SLM, small... and hence a full 100 pages PDF is not possible. The solution? Filter the PDF and find the chunks that are relevant and only use those.

> Slide: Tear some pages of a book and hand them over to a robot.

Hmm, interesting, so not all the PDF, just the relevant chunks... there must be an easier way to do this.

> Slide: RAG deployment

Indeed! You can use a database, pass the question to the database and get the chunks of documents relevant to the question.

Ok, so for this "talking to my data" to happen you will need:
- a vector database for the chunks
- an LLM (small in fact)
- a procedure to chunk the documents and put them into the DB
- a chat application

> Slide: Microwave dings!

Oops... the solution should be deployed already, let me have a look.


