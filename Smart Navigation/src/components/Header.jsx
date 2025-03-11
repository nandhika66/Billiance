import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <Link to="/" className="logo">
        <img src="/logo.png" alt="Murugan Stores Logo" className="logo-img" />
      </Link>
      <Link to="/login" className="login-btn">Login</Link>
    </header>
  );
};

export default Header;