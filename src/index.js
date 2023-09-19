import "dotenv/config";
import express from "express";
import sharp from "sharp";

const app = express();

import { HfInference } from "@huggingface/inference";

const HF_ACCESS_TOKEN = process.env.API;

const inference = new HfInference(HF_ACCESS_TOKEN);

//const model_id = "SG161222/Realistic_Vision_V1.4";
const model_id = "kandinsky-community/kandinsky-2-2-decoder";

const input = "portrait of a young women, blue eyes, cinematic in the florest";
const negative = "low quality, bad quality";

async function teste() {
  const image = await inference.textToImage({
    model: model_id,
    inputs: input,
    parameters: {
      negative_prompt: negative,
    },
  });

  const buffer = await image.arrayBuffer();
  const filename = `image${Math.random() * 100000}.png`;

  await sharp(buffer).toFile(filename);
  console.log("imagem gerada com sucesso!!!");
}

teste();

// You can also omit "model" to use the recommended model for the task

//   await inference.translation({
//     model: "t5-base",
//     inputs: "My name is Wolfgang and I live in Berlin",
//   })
// ;

// async function teste() {
//   const image = await inference.textToImage({
//     model: "stabilityai/stable-diffusion-2",
//     inputs:
//       "award winning high resolution photo of a giant tortoise/((ladybird)) hybrid, [trending on artstation]",
//     parameters: {
//       negative_prompt: "blurry",
//     },
//   });

//   // fs.appendFile("teste.jpg", image, function (err) {
//   //   if (err) throw err;
//   //   console.log("Saved!");
//   // });
//   console.log(image);
// }

// teste();

// const generateText = async () => {
//   const text = await inference.imageToText({
//     data: await (await fetch("https://picsum.photos/300/300")).blob(),
//     model: "nlpconnect/vit-gpt2-image-captioning",
//   });
//   console.log(text);
// };

// generateText();

// Using your own inference endpoint: https://hf.co/docs/inference-endpoints/
// const gpt2 = inference.endpoint(
//   "https://xyz.eu-west-1.aws.endpoints.huggingface.cloud/gpt2"
// );
// const { generated_text } = await gpt2.textGeneration({
//   inputs: "The answer to the universe is",
// });

app.listen(3000, () => console.log("Server Runing"));
