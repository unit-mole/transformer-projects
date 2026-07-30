interface ProjectFilterProps {
  category: string;
  deployment: string;
  onCategoryChange: (value: string) => void;
  onDeploymentChange: (value: string) => void;
}

const categories = ["All", "ANN", "Simple RNN", "LSTM", "BiLSTM", "CNN", "Transformer", "Portfolio"];
const deployments = ["All", "Hugging Face", "GitHub Pages", "Vercel", "Streamlit", "Gradio", "TensorFlow.js"];

export default function ProjectFilter({
  category,
  deployment,
  onCategoryChange,
  onDeploymentChange,
}: ProjectFilterProps) {
  return (
    <div className="filter-grid">
      <label>
        Project category
        <select value={category} onChange={(event) => onCategoryChange(event.target.value)}>
          {categories.map((option) => <option key={option}>{option}</option>)}
        </select>
      </label>
      <label>
        Deployment platform
        <select value={deployment} onChange={(event) => onDeploymentChange(event.target.value)}>
          {deployments.map((option) => <option key={option}>{option}</option>)}
        </select>
      </label>
    </div>
  );
}
