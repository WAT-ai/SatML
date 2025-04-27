'use client';

import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="flex justify-between items-center px-10 py-6 bg-black text-primary">
      <div className="text-2xl font-bold">SatML</div>
      <div className="flex space-x-8 text-sm font-light">
        <Link href="#">Home</Link>
        <Link href="#">About</Link>
        <Link href="#">Project</Link>
        <Link href="#">Contact</Link>
      </div>
    </nav>
  );
}
