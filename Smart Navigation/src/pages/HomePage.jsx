import React from 'react';
import { useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar';
import Carousel from '../components/Carousel';

const HomePage = () => {
  const navigate = useNavigate();

  const handleSearch = (query) => {
    navigate(`/item/${query}`);
  };

  
  const todaysDeals = [
    { name: 'Deal 1', price: 19.99, image: '/carousel_image2.jpg' },
    { name: 'Deal 2', price: 29.99, image: '/carousel_image1.jpg' },
    { name: 'Deal 3', price: 39.99, image: '/carousel_image3.jpg' },
    { name: 'Deal 4', price: 49.99, image: '/carousel_image1.jpg' },
  ];

  const bestSellers = [
    { name: 'Best Seller 1', price: 24.99, image: '/carousel_image2.jpg' },
    { name: 'Best Seller 2', price: 34.99, image: '/carousel_image3.jpg' },
    { name: 'Best Seller 3', price: 44.99, image: '/carousel_image1.jpg' },
    { name: 'Best Seller 4', price: 54.99, image: '/carousel_image2.jpg' },
  ];

  const newArrivals = [
    { name: 'New Arrival 1', price: 14.99, image: '/carousel_image3.jpg' },
    { name: 'New Arrival 2', price: 24.99, image: '/carousel_image2.jpg' },
    { name: 'New Arrival 3', price: 34.99, image: '/carousel_image1.jpg' },
    { name: 'New Arrival 4', price: 44.99, image: '/carousel_image2.jpg' },
  ];

  return (
    <div className="container">
      <SearchBar onSearch={handleSearch} />
      <Carousel title="Today's Deals" products={todaysDeals} />
      <Carousel title="Best Sellers" products={bestSellers} />
    </div>
  );
};

export default HomePage;



