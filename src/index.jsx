import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';
import { Route, BrowserRouter, Routes } from 'react-router-dom';
import PublicRoutes from './routes/PublicRoutes';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {PublicRoutes.map((page, index) => {
            return (
              <Route element = {<page.content/>} key={index} path={page.path}/>
            )
        })}
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

reportWebVitals();
