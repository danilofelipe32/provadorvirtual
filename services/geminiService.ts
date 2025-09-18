/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { GoogleGenAI, GenerateContentResponse, Modality } from "@google/genai";

const fileToPart = async (file: File) => {
    const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
    });
    const { mimeType, data } = dataUrlToParts(dataUrl);
    return { inlineData: { mimeType, data } };
};

const dataUrlToParts = (dataUrl: string) => {
    const arr = dataUrl.split(',');
    if (arr.length < 2) throw new Error("Invalid data URL");
    const mimeMatch = arr[0].match(/:(.*?);/);
    if (!mimeMatch || !mimeMatch[1]) throw new Error("Could not parse MIME type from data URL");
    return { mimeType: mimeMatch[1], data: arr[1] };
}

const dataUrlToPart = (dataUrl: string) => {
    const { mimeType, data } = dataUrlToParts(dataUrl);
    return { inlineData: { mimeType, data } };
}

const handleApiResponse = (response: GenerateContentResponse): string => {
    if (response.promptFeedback?.blockReason) {
        const { blockReason, blockReasonMessage } = response.promptFeedback;
        const errorMessage = `A solicitação foi bloqueada. Motivo: ${blockReason}. ${blockReasonMessage || ''}`;
        throw new Error(errorMessage);
    }

    // Find the first image part in any candidate
    for (const candidate of response.candidates ?? []) {
        const imagePart = candidate.content?.parts?.find(part => part.inlineData);
        if (imagePart?.inlineData) {
            const { mimeType, data } = imagePart.inlineData;
            return `data:${mimeType};base64,${data}`;
        }
    }

    const finishReason = response.candidates?.[0]?.finishReason;
    if (finishReason && finishReason !== 'STOP') {
        const errorMessage = `A geração de imagem parou inesperadamente. Motivo: ${finishReason}. Isso geralmente está relacionado às configurações de segurança.`;
        throw new Error(errorMessage);
    }
    const textFeedback = response.text?.trim();
    const errorMessage = `O modelo de IA não retornou uma imagem. ` + (textFeedback ? `O modelo respondeu com o texto: "${textFeedback}"` : "Isso pode acontecer devido a filtros de segurança ou se a solicitação for muito complexa. Por favor, tente uma imagem diferente.");
    throw new Error(errorMessage);
};

const ai = new GoogleGenAI({ apiKey: "AIzaSyC7KYUNng5ruXGovkb0Ky-QtUz3wEcFKPk" });
const model = 'gemini-2.5-flash-image-preview';

export const generateModelImage = async (userImage: File): Promise<string> => {
    const userImagePart = await fileToPart(userImage);
    const prompt = "Você é uma IA especialista em fotografia de moda. Transforme a pessoa nesta imagem em uma foto de modelo de corpo inteiro para um site de e-commerce. O fundo deve ser um estúdio neutro e limpo (cinza claro, #f0f0f0). A pessoa deve ter uma expressão de modelo neutra e profissional. Preserve a identidade da pessoa, características únicas e tipo de corpo, mas coloque-a em uma pose de modelo padrão, relaxada и em pé. A imagem final deve ser fotorrealista. Retorne APENAS a imagem final.";
    const response = await ai.models.generateContent({
        model,
        contents: { parts: [userImagePart, { text: prompt }] },
        config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
    });
    return handleApiResponse(response);
};

export const generateVirtualTryOnImage = async (modelImageUrl: string, garmentImage: File): Promise<string> => {
    const modelImagePart = dataUrlToPart(modelImageUrl);
    const garmentImagePart = await fileToPart(garmentImage);
    const prompt = `Você é uma IA especialista em provador virtual. Você receberá uma 'imagem da modelo' e uma 'imagem da peça de roupa'. Sua tarefa é criar uma nova imagem fotorrealista onde a pessoa da 'imagem da modelo' está vestindo a roupa da 'imagem da peça de roupa'.

**Regras Cruciais:**
1.  **Substituição Completa da Roupa:** Você DEVE remover e substituir COMPLETAMENTE a peça de roupa usada pela pessoa na 'imagem da modelo' pela nova peça. Nenhuma parte da roupa original (por exemplo, golas, mangas, estampas) deve ser visível na imagem final.
2.  **Preservar a Modelo:** O rosto, cabelo, formato do corpo e pose da pessoa da 'imagem da modelo' DEVEM permanecer inalterados.
3.  **Preservar o Fundo:** Todo o fundo da 'imagem da modelo' DEVE ser preservado perfeitamente.
4.  **Aplicar a Peça de Roupa:** Ajuste realisticamente a nova peça de roupa na pessoa. Ela deve se adaptar à pose com dobras, sombras e iluminação naturais consistentes com a cena original.
5.  **Saída:** Retorne APENAS a imagem final editada. Não inclua nenhum texto.`;
    const response = await ai.models.generateContent({
        model,
        contents: { parts: [modelImagePart, garmentImagePart, { text: prompt }] },
        config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
    });
    return handleApiResponse(response);
};

export const generatePoseVariation = async (tryOnImageUrl: string, poseInstruction: string): Promise<string> => {
    const tryOnImagePart = dataUrlToPart(tryOnImageUrl);
    const prompt = `Você é uma IA especialista em fotografia de moda. Pegue esta imagem e gere-a novamente de uma perspectiva diferente. A pessoa, a roupa e o estilo do fundo devem permanecer idênticos. A nova perspectiva deve ser: "${poseInstruction}". Retorne APENAS a imagem final.`;
    const response = await ai.models.generateContent({
        model,
        contents: { parts: [tryOnImagePart, { text: prompt }] },
        config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
    });
    return handleApiResponse(response);
};