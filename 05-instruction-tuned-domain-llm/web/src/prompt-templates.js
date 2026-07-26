const SYSTEM_SCOPE = [
  'You are an educational ML and Data Science learning assistant.',
  'Answer only within machine learning, data science, analytics, NLP, computer vision, MLOps, and quality analytics.',
  'Be accurate, beginner-friendly, structured, and explicit about uncertainty.',
  'Do not give legal, medical, financial, immigration, or safety-critical advice.',
].join(' ');

export const PROMPT_CATEGORIES = Object.freeze([
  'Concept explanation',
  'Algorithm comparison',
  'Metric explanation',
  'Example generation',
  'Interview-style answer',
  'Small code or pseudo-code example',
  'Data Science workflow',
  'Quality analytics example',
]);

const CATEGORY_INSTRUCTIONS = Object.freeze({
  'Concept explanation': 'Explain the concept clearly, define important terms, and include one intuitive example.',
  'Algorithm comparison': 'Compare the methods using assumptions, strengths, limitations, and suitable use cases.',
  'Metric explanation': 'Define the metric, explain when it is useful, and include a practical interpretation.',
  'Example generation': 'Generate a small, concrete educational example and explain each step.',
  'Interview-style answer': 'Give a concise interview-ready answer followed by one deeper technical note.',
  'Small code or pseudo-code example': 'Provide a short, safe code or pseudo-code example and explain what it demonstrates.',
  'Data Science workflow': 'Describe the workflow in ordered stages, including validation and evaluation.',
  'Quality analytics example': 'Connect the explanation to defect, case, process, supplier, or quality trend analysis without using confidential data.',
});

export function buildPrompt({ category, instruction, context = '' }) {
  const cleanInstruction = instruction.trim();
  const cleanContext = context.trim();
  const categoryGuidance = CATEGORY_INSTRUCTIONS[category] ?? CATEGORY_INSTRUCTIONS['Concept explanation'];

  return [
    `Context: ${SYSTEM_SCOPE}`,
    `Task style: ${categoryGuidance}`,
    '',
    'Instruction:',
    cleanInstruction,
    '',
    'Input:',
    cleanContext || 'No additional input provided.',
    '',
    'Response:',
  ].join('\n');
}
