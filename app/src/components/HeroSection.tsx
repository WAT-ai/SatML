'use client';

import Image from 'next/image';

export default function HeroSection() {
  return (
    <section className="relative w-full h-[85vh] bg-black text-primary overflow-hidden">
      <div className="relative flex flex-row items-center justify-between max-w-7xl mx-auto h-full px-10">
        
        {/* Image on the left */}
        <div className="w-1/2 h-full relative">
          <Image
            src="/earth.jpeg"
            alt="Earth"
            fill
            className="object-cover transform"
          />
        </div>

        {/* Text on the right */}
        <div className="w-1/2 flex flex-col justify-center items-start space-y-6 px-8">
          <h1 className="text-3xl font-normal max-w-4xl leading-tight">
            AI-powered methane emissions detection via hyper-spectral satellite imagery.
          </h1>
          <p className="text-gray-400 text-lg font-extralight italic">
            Analyze. Predict. Prevent.
          </p>
          <button className="mt-4 px-6 py-2 border border-white text-white rounded-md hover:bg-white hover:text-black transition">
            Learn More
          </button>
        </div>

      </div>
    </section>
  );
}
