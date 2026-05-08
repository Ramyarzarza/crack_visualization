import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import Labeler from './Labeler'
import './App.css'

function Root() {
  const [page, setPage] = React.useState(() =>
    window.location.hash === '#/label' ? 'label' : 'predict'
  )

  React.useEffect(() => {
    const onHash = () => setPage(window.location.hash === '#/label' ? 'label' : 'predict')
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return (
    <div className="app">
      <nav className="top-nav">
        <span className="top-nav-brand">Crack Detection</span>
        <div className="top-nav-links">
          <a
            href="#/predict"
            className={`top-nav-link ${page === 'predict' ? 'top-nav-link-active' : ''}`}
            onClick={() => setPage('predict')}
          >
            Inference
          </a>
          <a
            href="#/label"
            className={`top-nav-link ${page === 'label' ? 'top-nav-link-active' : ''}`}
            onClick={() => setPage('label')}
          >
            Labeling
          </a>
        </div>
      </nav>
      {page === 'predict' ? <App embedded /> : <Labeler />}
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
)

